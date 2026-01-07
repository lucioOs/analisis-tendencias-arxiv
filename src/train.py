from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from utils import Paths, log, die, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Proporción para test (temporal)")
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    P = Paths()
    P.ensure()

    inp = P.processed / "features.parquet"
    if not inp.exists():
        die("No existe data/processed/features.parquet. Ejecuta primero: python src/features.py")

    df = pd.read_parquet(inp).copy()
    if df.empty:
        die("features.parquet está vacío.")

    df = df.sort_values("period").copy()

    feat_cols = [
        "lag_1", "lag_2", "lag_3",
        "ma_3", "ma_5", "std_5",
        "slope_1", "growth", "count"
    ]

    X = df[feat_cols].values
    y = df["y"].values

    split = int(len(df) * (1.0 - args.test_ratio))
    if split < 100:
        log("AVISO: Split pequeño. Aun así continuaré.")

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    out_model = P.models / "model.pkl"
    joblib.dump({"model": model, "features": feat_cols}, out_model)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "n_estimators": args.n_estimators,
        "random_state": args.random_state
    }
    write_json(P.models / "metrics.json", metrics)

    log(f"OK: creado {out_model}")
    log(f"MAE:  {mae:.8f}")
    log(f"RMSE: {rmse:.8f}")


if __name__ == "__main__":
    main()
