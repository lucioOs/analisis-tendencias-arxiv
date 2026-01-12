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
        return pd.DataFrame(columns=["date", "text", "source"])

    cols = {c.lower(): c for c in df.columns}

    date_col = None
    for k in ["date", "published", "published_at", "updated", "created"]:
        if k in cols:
            date_col = cols[k]
            break

    text_col = None
    for k in ["text", "summary", "abstract", "content", "title"]:
        if k in cols:
            text_col = cols[k]
            break

    source_col = cols.get("source", None)

    if date_col is None or text_col is None:
        return pd.DataFrame(columns=["date", "text", "source"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out["text"] = df[text_col].astype(str)

    if source_col:
        out["source"] = df[source_col].astype(str)
    else:
        out["source"] = default_source

    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 20]
    return out


def main():
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
        print("[WARN] No hay datos para exportar. (No existe live_arxiv.parquet o está vacío)")
        return 2

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "text", "source"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    unique_days = df["date"].dt.date.nunique()
    print(f"[INFO] Exportado: {OUT} | rows={len(df)} | min={df['date'].min()} | max={df['date'].max()} | unique_days={unique_days}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
