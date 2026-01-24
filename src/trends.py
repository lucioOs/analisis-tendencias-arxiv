from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from utils import Paths, log, die


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="W", choices=["W", "M"], help="Frecuencia: W=semana, M=mes")
    ap.add_argument("--min_df", type=int, default=20, help="Mínimo de documentos para incluir término")
    ap.add_argument("--ngram_max", type=int, default=2, help="n-gram máximo (1-3 recomendado)")
    ap.add_argument("--top_k", type=int, default=50, help="Top K por periodo (según growth)")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / "clean.parquet"
    if not inp.exists():
        die("No existe data/processed/clean.parquet. Ejecuta primero: python src/preprocess.py")

    df = pd.read_parquet(inp).copy()
    if df.empty:
        die("clean.parquet está vacío.")

    # Para tendencias temporales necesitas fecha
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        die("No hay fechas válidas (date) para agrupar por tiempo. Revisa dataset_schema.json.")

    df["period"] = df["date"].dt.to_period(args.freq).dt.start_time
    log(f"Periodos: {df['period'].min()} -> {df['period'].max()} | filas={len(df)}")

    vec = CountVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df
    )

    X = vec.fit_transform(df["text_clean"])
    terms = np.array(vec.get_feature_names_out())

    # Conteo por periodo
    rows = []
    grouped = df.groupby("period").groups
    for p, idx in grouped.items():
        Xp = X[idx]
        counts = np.asarray(Xp.sum(axis=0)).ravel()
        total = counts.sum() if counts.sum() > 0 else 1
        rel = counts / total

        part = pd.DataFrame({
            "period": p,
            "term": terms,
            "count": counts.astype(int),
            "rel_freq": rel.astype(float),
        })
        rows.append(part)

    ts = pd.concat(rows, ignore_index=True)
    ts = ts[ts["count"] > 0].copy()
    ts = ts.sort_values(["term", "period"])

    # Growth score: cambio relativo vs periodo anterior (sobre rel_freq)
    ts["rel_prev"] = ts.groupby("term")["rel_freq"].shift(1)
    ts["growth"] = (ts["rel_freq"] - ts["rel_prev"]) / (ts["rel_prev"].replace(0, np.nan))
    ts["growth"] = ts["growth"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Top por periodo
    ts["rank_growth"] = ts.groupby("period")["growth"].rank(ascending=False, method="first")
    top = ts[ts["rank_growth"] <= args.top_k].copy()

    out = P.processed / "trends.parquet"
    top.to_parquet(out, index=False)

    log(f"OK: creado {out} shape={top.shape}")
    log("Ejemplo (último periodo):")
    last = top["period"].max()
    sample = top[top["period"] == last].sort_values("growth", ascending=False).head(10)
    log(sample[["term", "count", "rel_freq", "growth"]].to_string(index=False))


if __name__ == "__main__":
    main()
