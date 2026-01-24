from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="API - Tendencias (PLN + ML)")

TRENDS = Path("data/processed/trends.parquet")
FEATURES = Path("data/processed/features.parquet")
MODEL = Path("models/model.pkl")


def require_file(path: Path) -> None:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"No existe archivo requerido: {path}")


class PredictRequest(BaseModel):
    term: str = Field(..., min_length=1, description="Término (n-gram) a predecir")
    top_k_fallback: int = Field(30, ge=1, le=200, description="Si el término no existe, sugerir términos del Top")


class PredictResponse(BaseModel):
    term: str
    predicted_rel_freq_next_period: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trends")
def trends(top_k: int = 20):
    require_file(TRENDS)
    df = pd.read_parquet(TRENDS)
    if df.empty:
        raise HTTPException(status_code=500, detail="trends.parquet está vacío")

    latest = df["period"].max()
    out = (
        df[df["period"] == latest]
        .sort_values("growth", ascending=False)
        .head(top_k)[["period", "term", "count", "rel_freq", "growth"]]
    )
    return out.to_dict(orient="records")


@app.get("/terms")
def terms(limit: int = 200) -> List[str]:
    """Lista términos disponibles para predicción (desde features)."""
    require_file(FEATURES)
    df = pd.read_parquet(FEATURES, columns=["term"])
    terms = sorted(df["term"].unique().tolist())
    return terms[:limit]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    require_file(FEATURES)
    require_file(MODEL)

    pack = joblib.load(MODEL)
    model = pack["model"]
    feat_cols = pack["features"]

    df = pd.read_parquet(FEATURES)
    df_term = df[df["term"] == req.term].sort_values("period")

    if df_term.empty:
        # sugerencias útiles
        require_file(TRENDS)
        tdf = pd.read_parquet(TRENDS)
        latest = tdf["period"].max()
        sugg = (tdf[tdf["period"] == latest]
                .sort_values("growth", ascending=False)
                .head(req.top_k_fallback)["term"]
                .tolist())
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Término no encontrado para predicción.",
                "hint": "Usa GET /terms para ver términos disponibles o elige uno de sugerencias.",
                "suggestions": sugg
            }
        )

    x_last = df_term.iloc[-1][feat_cols].values.reshape(1, -1)
    y_hat = float(model.predict(x_last)[0])

    return {"term": req.term, "predicted_rel_freq_next_period": y_hat}
