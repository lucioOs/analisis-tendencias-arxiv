from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utils import Paths, log, die


URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NONWORD_RE = re.compile(r"[^a-z0-9áéíóúñü\s]+", re.IGNORECASE)


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    # #GenerativeAI -> GenerativeAI (se queda como término)
    t = HASHTAG_RE.sub(r"\1", t)
    t = NONWORD_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_len", type=int, default=10, help="Longitud mínima del texto limpio")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / "dataset.parquet"
    if not inp.exists():
        die("No existe data/processed/dataset.parquet. Ejecuta primero: python src/load_data.py")

    df = pd.read_parquet(inp)
    if df.empty:
        die("dataset.parquet está vacío.")

    log(f"Leído {inp} shape={df.shape}")

    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() >= args.min_len].copy()

    out = P.processed / "clean.parquet"
    df.to_parquet(out, index=False)

    log(f"OK: creado {out} shape={df.shape}")
    log(f"Fechas nulas: {df['date'].isna().mean():.3f}")


if __name__ == "__main__":
    main()
