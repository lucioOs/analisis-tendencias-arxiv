from __future__ import annotations

import argparse
import pandas as pd

from utils import Paths, log, die


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, help="Cuántos periodos adelante predecir")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / "trends.parquet"
    if not inp.exists():
        die("No existe data/processed/trends.parquet. Ejecuta primero: python src/trends.py")

    df = pd.read_parquet(inp).copy()
    if df.empty:
        die("trends.parquet está vacío.")

    df = df.sort_values(["term", "period"]).copy()

    # target: rel_freq futuro
    df["y"] = df.groupby("term")["rel_freq"].shift(-args.horizon)

    # lags
    for k in [1, 2, 3]:
        df[f"lag_{k}"] = df.groupby("term")["rel_freq"].shift(k)

    # rolling stats
    df["ma_3"] = df.groupby("term")["rel_freq"].rolling(3).mean().reset_index(level=0, drop=True)
    df["ma_5"] = df.groupby("term")["rel_freq"].rolling(5).mean().reset_index(level=0, drop=True)
    df["std_5"] = df.groupby("term")["rel_freq"].rolling(5).std().reset_index(level=0, drop=True)
    df["slope_1"] = df["rel_freq"] - df["lag_1"]

    feat_cols = [
        "lag_1", "lag_2", "lag_3",
        "ma_3", "ma_5", "std_5",
        "slope_1", "growth", "count"
    ]

    out = df.dropna(subset=feat_cols + ["y"]).copy()

    if out.empty:
        die(
            "No se pudieron crear features (dataset quedó vacío). "
            "Prueba bajar --min_df en trends.py o usar freq=M para agrupar por mes."
        )

    out_path = P.processed / "features.parquet"
    out.to_parquet(out_path, index=False)

    log(f"OK: creado {out_path} shape={out.shape}")
    log(f"Terms únicos con features: {out['term'].nunique()}")


if __name__ == "__main__":
    main()
