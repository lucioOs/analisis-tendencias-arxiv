# src/config.py
from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------
# Base paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# -----------------------------
# App UI
# -----------------------------
APP_TITLE = os.getenv("APP_TITLE", "Panel de Tendencias en Computación")

# -----------------------------
# Cache / performance
# -----------------------------
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
MAX_ROWS_TEXT = int(os.getenv("MAX_ROWS_TEXT", "7000"))

# -----------------------------
# Histórico: conservar N años
# -----------------------------
HIST_YEARS_KEEP = int(os.getenv("HIST_YEARS_KEEP", "10"))

# -----------------------------
# Datasets (parquet)
# -----------------------------
HIST_DATASET = Path(os.getenv("HIST_DATASET", str(PROCESSED_DIR / "dataset.parquet")))
LIVE_DATASET = Path(os.getenv("LIVE_DATASET", str(PROCESSED_DIR / "live_dataset.parquet")))
LIVE_ARXIV = Path(os.getenv("LIVE_ARXIV", str(PROCESSED_DIR / "live_arxiv.parquet")))

# -----------------------------
# Runner live (recomendado: módulo)
# -----------------------------
# RECOMENDADO para evitar "No module named 'src'":
#   python -m src.live_runner
LIVE_RUNNER_MODULE = [sys.executable, "-m", "src.live_runner"]

# Fallback si alguna vez lo necesitas como archivo (no recomendado)
LIVE_RUNNER_SCRIPT = [sys.executable, str(PROJECT_ROOT / "src" / "live_runner.py")]

# Mantén este alias para compatibilidad con código viejo:
# (apunta al método recomendado)
LIVE_RUNNER = LIVE_RUNNER_MODULE

# -----------------------------
# Agregados macro-áreas (opcionales)
# -----------------------------
AREA_TS = Path(os.getenv("AREA_TS", str(PROCESSED_DIR / "area_ts.parquet")))
AREA_TOP = Path(os.getenv("AREA_TOP", str(PROCESSED_DIR / "area_top.parquet")))

# -----------------------------
# Artefactos extra (opcionales)
# -----------------------------
TRENDS_PARQUET = Path(os.getenv("TRENDS_PARQUET", str(PROCESSED_DIR / "trends.parquet")))
TRENDS_FORECAST = Path(os.getenv("TRENDS_FORECAST", str(PROCESSED_DIR / "trends_forecast.parquet")))

# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ensure_dirs()
