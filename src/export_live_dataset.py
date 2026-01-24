from __future__ import annotations

from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

ARXIV = PROCESSED / "live_arxiv.parquet"
RSS = PROCESSED / "live_rss.parquet"
OUT = PROCESSED / "live_dataset.parquet"


def _normalize(df: pd.DataFrame, default_source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "text", "source", "id", "url", "title"])

    cols = {c.lower(): c for c in df.columns}

    date_col = next((cols[k] for k in ["date", "published", "published_at", "updated", "created"] if k in cols), None)
    text_col = next((cols[k] for k in ["text", "summary", "abstract", "content"] if k in cols), None)

    # si no hay "text", armamos con title+summary si existen
    title_col = cols.get("title")
    if text_col is None and title_col is not None:
        text_col = title_col

    source_col = cols.get("source")
    id_col = cols.get("id")
    url_col = cols.get("url") or cols.get("link")

    if date_col is None or text_col is None:
        return pd.DataFrame(columns=["date", "text", "source", "id", "url", "title"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)

    # texto: si hay summary y title, combinamos (se ve mejor en dashboard)
    if title_col is not None and "summary" in cols:
        t = df[title_col].astype(str)
        s = df[cols["summary"]].astype(str)
        out["text"] = (t.str.strip() + ". " + s.str.strip()).str.strip()
        out["title"] = t.astype(str)
    else:
        out["text"] = df[text_col].astype(str)
        out["title"] = df[title_col].astype(str) if title_col is not None else ""

    out["source"] = df[source_col].astype(str) if source_col else default_source
    out["id"] = df[id_col].astype(str) if id_col else ""
    out["url"] = df[url_col].astype(str) if url_col else ""

    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 20]
    return out


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Primero: si hay id o url, dedup fuerte
    if df["id"].astype(str).str.len().gt(3).any():
        df = df.drop_duplicates(subset=["source", "id"], keep="last")
    elif df["url"].astype(str).str.len().gt(5).any():
        df = df.drop_duplicates(subset=["source", "url"], keep="last")
    else:
        df = df.drop_duplicates(subset=["source", "date", "text"], keep="last")

    return df


def main() -> int:
    parts = []

    if ARXIV.exists():
        df_a = pd.read_parquet(ARXIV)
        df_a = _normalize(df_a, "arxiv")
        if not df_a.empty:
            parts.append(df_a)

    if RSS.exists():
        df_r = pd.read_parquet(RSS)
        df_r = _normalize(df_r, "rss")
        if not df_r.empty:
            parts.append(df_r)

    if not parts:
        print("[WARN] No hay datos para exportar.")
        return 2

    df = pd.concat(parts, ignore_index=True)
    df = _dedup(df)
    df = df.sort_values("date").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    unique_days = df["date"].dt.date.nunique()
    print(f"[INFO] Exportado: {OUT} | rows={len(df)} | min={df['date'].min()} | max={df['date'].max()} | unique_days={unique_days}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
