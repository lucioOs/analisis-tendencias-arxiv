# src/export_live_dataset.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


PROCESSED = Path("data/processed")
LIVE_OUT = PROCESSED / "live_dataset.parquet"

# Ajusta estos paths si tus ingestas guardan en otro lugar
ARXIV_DB = PROCESSED / "live_arxiv.parquet"
RSS_DB = PROCESSED / "live_rss.parquet"

def log(level: str, msg: str) -> None:
    # Evita problemas de encoding en Windows
    safe = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    print(f"[{level}] {safe}", flush=True)

def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        log("WARN", f"No se pudo leer {path}: {e}")
        return pd.DataFrame()

def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df

    # Normaliza nombres esperados
    # Debe existir: date, text. Si no, intenta inferir.
    cols = {c.lower(): c for c in df.columns}

    # Detectar texto
    text_col = None
    for k in ["text", "content", "summary", "title", "abstract", "body"]:
        if k in cols:
            text_col = cols[k]
            break
    if text_col is None:
        # último recurso: concat de columnas string
        text_candidates = [c for c in df.columns if df[c].dtype == object]
        if not text_candidates:
            return pd.DataFrame()
        df["_text_join"] = df[text_candidates].astype(str).agg(" ".join, axis=1)
        text_col = "_text_join"

    # Detectar fecha
    date_col = None
    for k in ["date", "published", "published_at", "updated", "created"]:
        if k in cols:
            date_col = cols[k]
            break
    if date_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out["text"] = df[text_col].astype(str)
    out["source"] = source

    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 20]

    return out

def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    log("INFO", f"Leyendo fuentes locales: {ARXIV_DB.name}, {RSS_DB.name}")

    df_arxiv = normalize(read_optional(ARXIV_DB), "arxiv")
    df_rss = normalize(read_optional(RSS_DB), "rss")

    df = pd.concat([df_arxiv, df_rss], ignore_index=True)

    if df.empty:
        log("WARN", "No hay datos para exportar. live_dataset.parquet no se generara.")
        return 2

    # Deduplicado básico
    df["text_hash"] = df["text"].str.lower().str.slice(0, 300)
    df = df.drop_duplicates(subset=["text_hash"]).drop(columns=["text_hash"])
    df = df.sort_values("date").reset_index(drop=True)

    df.to_parquet(LIVE_OUT, index=False)

    log("INFO", f"Exportado: {LIVE_OUT}")
    log("INFO", f"Filas: {len(df)}")
    log("INFO", f"Rango: {df['date'].min()} -> {df['date'].max()}")
    log("INFO", f"Fuentes: {df['source'].value_counts().to_dict()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
