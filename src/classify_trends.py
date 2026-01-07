from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


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


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def classify_term(row: pd.Series,
                  vol_p70: float,
                  vol_p30: float,
                  growth_p70: float,
                  growth_p30: float) -> str:
    """
    Clasificación robusta basada en volumen y crecimiento reciente.
    - volumen: promedio rel_freq en ventana reciente
    - crecimiento: mediana de growth en ventana reciente
    """
    vol = row["vol_recent"]
    gr = row["growth_recent"]
    slope = row["slope_recent"]

    # Declive: crecimiento bajo + pendiente negativa
    if gr <= growth_p30 and slope < 0:
        return "declive"

    # Emergente: crecimiento alto + volumen aún bajo/medio
    if gr >= growth_p70 and vol <= vol_p70:
        return "emergente"

    # Consolidada: volumen alto y crecimiento no negativo fuerte (estable)
    if vol >= vol_p70 and gr >= growth_p30:
        return "consolidada"

    # Neutra: no destaca
    return "neutra"


def main() -> None:
    ap = argparse.ArgumentParser(description="Clasificación de tendencias (emergente/consolidada/declive)")
    ap.add_argument("--input", default="trends_full.parquet", help="Entrada en data/processed")
    ap.add_argument("--freq_window", type=int, default=6, help="Ventana reciente en periodos (meses si freq=M)")
    ap.add_argument("--min_periods", type=int, default=8, help="Mínimo de periodos totales para evaluar un término")
    ap.add_argument("--output", default="trend_classes.parquet", help="Salida en data/processed")
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / args.input
    if not inp.exists():
        die(f"No existe {inp}. Ejecuta: python src/trends_full.py")

    df = pd.read_parquet(inp)
    if df.empty:
        die(f"{inp.name} está vacío")

    needed = {"period", "term", "rel_freq", "growth"}
    if not needed.issubset(set(df.columns)):
        die(f"Faltan columnas requeridas en {inp.name}. Necesito: {sorted(needed)}")

    df = df.sort_values(["term", "period"]).copy()

    # features por término
    g = df.groupby("term", as_index=False)

    # conteo de periodos
    periods_count = df.groupby("term")["period"].nunique().rename("n_periods")
    df = df.merge(periods_count, on="term", how="left")

    # filtrar términos con muy poca historia
    terms_ok = df[df["n_periods"] >= args.min_periods]["term"].unique()
    df_ok = df[df["term"].isin(terms_ok)].copy()
    if df_ok.empty:
        die(
            "No hay términos suficientes con historia para clasificar. "
            "Baja --min_periods o usa freq=M con más periodos."
        )

    # Ventana reciente por término (últimos N periodos)
    def recent_stats(x: pd.DataFrame) -> pd.Series:
        x = x.sort_values("period")
        recent = x.tail(args.freq_window)
        vol_recent = float(recent["rel_freq"].mean())
        growth_recent = float(recent["growth"].median())
        # slope simple: ultimo - primero en ventana
        slope_recent = float(recent["rel_freq"].iloc[-1] - recent["rel_freq"].iloc[0])
        last_period = recent["period"].iloc[-1]
        return pd.Series({
            "last_period": last_period,
            "vol_recent": vol_recent,
            "growth_recent": growth_recent,
            "slope_recent": slope_recent,
            "n_periods": int(x["period"].nunique()),
        })

    stats = df_ok.groupby("term").apply(recent_stats).reset_index()

    # Umbrales robustos por percentiles (evita reglas arbitrarias)
    vol_p70 = float(stats["vol_recent"].quantile(0.70))
    vol_p30 = float(stats["vol_recent"].quantile(0.30))
    growth_p70 = float(stats["growth_recent"].quantile(0.70))
    growth_p30 = float(stats["growth_recent"].quantile(0.30))

    stats["class"] = stats.apply(
        lambda r: classify_term(r, vol_p70, vol_p30, growth_p70, growth_p30),
        axis=1
    )

    # Score de prioridad (para ranking): crecimiento + volumen normalizados
    stats["priority_score"] = (
        (stats["growth_recent"] - stats["growth_recent"].min()) /
        (stats["growth_recent"].max() - stats["growth_recent"].min() + 1e-9)
        +
        (stats["vol_recent"] - stats["vol_recent"].min()) /
        (stats["vol_recent"].max() - stats["vol_recent"].min() + 1e-9)
    )

    outp = P.processed / args.output
    stats = stats.sort_values(["class", "priority_score"], ascending=[True, False])
    stats.to_parquet(outp, index=False)

    meta = {
        "input": inp.name,
        "output": outp.name,
        "freq_window": args.freq_window,
        "min_periods": args.min_periods,
        "thresholds": {
            "vol_p70": vol_p70,
            "vol_p30": vol_p30,
            "growth_p70": growth_p70,
            "growth_p30": growth_p30
        },
        "counts": stats["class"].value_counts().to_dict()
    }
    write_json(P.processed / "trend_classes_meta.json", meta)

    log("INFO", f"✅ creado {outp} | terms clasificados={len(stats)}")
    log("INFO", f"Meta: {P.processed / 'trend_classes_meta.json'}")

    # muestra rápida por clase
    for c in ["emergente", "consolidada", "declive", "neutra"]:
        top = stats[stats["class"] == c].head(10)[["term", "vol_recent", "growth_recent", "slope_recent", "priority_score"]]
        log("INFO", f"\nTop {c}:")
        if top.empty:
            log("INFO", "  (sin resultados)")
        else:
            log("INFO", top.to_string(index=False))


if __name__ == "__main__":
    main()
