from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# -----------------------------
# Logging / errores
# -----------------------------
def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log("ERROR", msg)
    raise SystemExit(code)


@dataclass(frozen=True)
class Paths:
    processed: Path = Path("data/processed")

    def ensure(self) -> None:
        self.processed.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tendencias FULL por periodo (sin top-k)")
    ap.add_argument("--freq", default="M", choices=["W", "M"], help="Frecuencia de periodo")
    ap.add_argument("--min_df", type=int, default=3, help="Mínimo de documentos para incluir término")
    ap.add_argument("--ngram_max", type=int, default=2, help="n-gram máximo")
    ap.add_argument("--input", default="clean.parquet", help="Archivo de entrada en data/processed")
    ap.add_argument("--output", default="trends_full.parquet", help="Salida en data/processed")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / args.input
    if not inp.exists():
        die(f"No existe {inp}. Ejecuta: python src/preprocess.py")

    df = pd.read_parquet(inp)
    if df.empty:
        die(f"{inp.name} está vacío")

    # Requiere fecha
    if "date" not in df.columns:
        die("No existe columna 'date' en clean.parquet")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        die("No hay fechas válidas para agrupar por periodos")

    if "text_clean" not in df.columns:
        die("No existe columna 'text_clean'. Revisa preprocess.py")

    df["period"] = df["date"].dt.to_period(args.freq).dt.start_time
    log("INFO", f"Periodos: {df['period'].min()} -> {df['period'].max()} | filas={len(df)}")

    vec = CountVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df
    )

    X = vec.fit_transform(df["text_clean"])
    terms = np.array(vec.get_feature_names_out())

    # Conteo por periodo
    rows: List[pd.DataFrame] = []
    groups = df.groupby("period").groups
    for p, idx in groups.items():
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
        part = part[part["count"] > 0]
        rows.append(part)

    full = pd.concat(rows, ignore_index=True)
    full = full.sort_values(["term", "period"]).reset_index(drop=True)

    # Growth por periodo
    full["rel_prev"] = full.groupby("term")["rel_freq"].shift(1)
    full["growth"] = (full["rel_freq"] - full["rel_prev"]) / (full["rel_prev"].replace(0, np.nan))
    full["growth"] = full["growth"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    outp = P.processed / args.output
    full.to_parquet(outp, index=False)

    meta = {
        "freq": args.freq,
        "min_df": args.min_df,
        "ngram_max": args.ngram_max,
        "input": inp.name,
        "output": outp.name,
        "n_terms": int(full["term"].nunique()),
        "n_rows": int(len(full)),
        "period_min": str(full["period"].min()),
        "period_max": str(full["period"].max()),
    }
    write_json(P.processed / "trends_full_meta.json", meta)

    log("INFO", f"✅ creado {outp} | filas={len(full)} | terms={full['term'].nunique()}")
    log("INFO", f"Meta: {P.processed / 'trends_full_meta.json'}")


if __name__ == "__main__":
    main()
