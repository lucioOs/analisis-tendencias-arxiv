from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import Paths, log, die, write_json
from taxonomy import map_term_to_area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_terms", type=int, default=25, help="Top términos por área y periodo")
    ap.add_argument("--outdir", default="data_public/aggregates", help="Salida de agregados")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / "trends.parquet"
    if not inp.exists():
        die("No existe data/processed/trends.parquet. Ejecuta primero: python src/trends.py")

    df = pd.read_parquet(inp).copy()
    if df.empty:
        die("trends.parquet está vacío.")

    # columnas mínimas esperadas
    needed = {"period", "term", "count", "rel_freq", "growth"}
    missing = needed - set(df.columns)
    if missing:
        die(f"trends.parquet no tiene columnas requeridas: {sorted(missing)}")

    # asignación de área
    df["area"] = df["term"].astype(str).map(map_term_to_area)

    # agregado por área y periodo
    area_ts = (
        df.groupby(["period", "area"], as_index=False)
          .agg(
              count=("count", "sum"),
              rel_freq=("rel_freq", "sum"),
              growth=("growth", "mean"),
              n_terms=("term", "nunique"),
          )
          .sort_values(["period", "count"], ascending=[True, False])
    )

    # top términos por área y periodo (por crecimiento)
    df2 = df.copy()
    df2["rank"] = df2.groupby(["period", "area"])["growth"].rank(ascending=False, method="first")
    area_top = df2[df2["rank"] <= args.top_terms].copy()
    area_top = area_top.sort_values(["period", "area", "growth"], ascending=[True, True, False])

    # guardar en data_public/aggregates
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p_ts = outdir / "area_timeseries.parquet"
    p_top = outdir / "area_top_terms.parquet"
    area_ts.to_parquet(p_ts, index=False)
    area_top.to_parquet(p_top, index=False)

    meta = {
        "source": str(inp),
        "rows": int(len(df)),
        "periods": int(df["period"].nunique()),
        "terms": int(df["term"].nunique()),
        "areas": sorted(df["area"].unique().tolist()),
        "top_terms_per_area_period": int(args.top_terms),
    }
    write_json(outdir / "meta_areas.json", meta)

    log(f"OK: {p_ts} shape={area_ts.shape}")
    log(f"OK: {p_top} shape={area_top.shape}")
    log(f"OK: {outdir / 'meta_areas.json'}")


if __name__ == "__main__":
    main()
