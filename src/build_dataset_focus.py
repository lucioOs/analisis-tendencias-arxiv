# src/build_dataset_focus.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _parse_date(s: pd.Series) -> pd.Series:
    # Acepta datetime o string; regresa datetime naive (sin tz) consistente
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def _normalize_categories(col: pd.Series) -> pd.Series:
    # categories puede venir None, lista o string
    def to_str(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " ".join(str(v) for v in x)
        return str(x)
    return col.map(to_str).fillna("")


def main() -> int:
    ap = argparse.ArgumentParser(description="Crea un dataset histórico 'focus' a partir del parquet grande.")
    ap.add_argument("--inp", default="data/processed/dataset.parquet", help="Parquet base (Kaggle procesado)")
    ap.add_argument("--out", default="data/processed/dataset_focus.parquet", help="Parquet 'focus' de salida")

    # filtros del proyecto (puedes ajustar sin tocar código)
    ap.add_argument(
        "--cats",
        default="cs.AI,cs.LG,cs.CL,stat.ML",
        help="Categorías permitidas separadas por coma (ej: cs.AI,cs.LG,cs.CL,stat.ML)",
    )
    ap.add_argument("--since", default="2012-01-01", help="Recorta desde esta fecha (YYYY-MM-DD) para acelerar")
    ap.add_argument("--min_text_len", type=int, default=50, help="Longitud mínima de texto (title+abstract)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"[ERROR] No existe: {inp}")
        return 2

    df = pd.read_parquet(inp)

    # Validar columnas mínimas
    for need in ["date", "text"]:
        if need not in df.columns:
            print(f"[ERROR] Falta columna '{need}' en {inp}. Columnas: {list(df.columns)}")
            return 3

    # Normalizar date/text
    df = df.copy()
    df["date"] = _parse_date(df["date"])
    df["text"] = df["text"].astype(str).fillna("").str.strip()

    df = df.dropna(subset=["date"])
    df = df[df["text"].str.len() >= int(args.min_text_len)]

    # Filtro por categorías si existe la columna
    allow = [c.strip() for c in str(args.cats).split(",") if c.strip()]
    if allow and "categories" in df.columns:
        cats = _normalize_categories(df["categories"])
        # match por tokens (evita falsos positivos)
        pattern = r"(^|\s)(" + "|".join(map(lambda x: x.replace(".", r"\."), allow)) + r")(\s|$)"
        df = df[cats.str.contains(pattern, regex=True, na=False)].copy()
    elif allow and "categories" not in df.columns:
        print("[WARN] No existe columna 'categories'. Se omite filtro por categoría.")

    # Recorte por fecha (acelera)
    since = pd.to_datetime(args.since, errors="coerce")
    if pd.notna(since):
        df = df[df["date"] >= since].copy()

    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(out, index=False)

    print(f"[OK] {out} | rows={len(df):,} | min={df['date'].min()} | max={df['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
