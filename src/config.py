# src/config.py
from __future__ import annotations

import os
import sys
from pathlib import Path

# =============================================================================
# Base paths
# =============================================================================
# Raíz del proyecto (…/Proyecto)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Carpeta de datos
DATA_DIR = PROJECT_ROOT / "data"

# Subcarpetas por responsabilidad
PROCESSED_DIR = DATA_DIR / "processed"   # histórico + artefactos
LIVE_DIR = DATA_DIR / "live"              # live_runner (LIVE real)

# =============================================================================
# App UI
# =============================================================================
APP_TITLE = os.getenv("APP_TITLE", "Panel de Tendencias en Computación")

# =============================================================================
# Cache / performance
# =============================================================================
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
MAX_ROWS_TEXT = int(os.getenv("MAX_ROWS_TEXT", "7000"))

# =============================================================================
# Histórico
# =============================================================================
HIST_YEARS_KEEP = int(os.getenv("HIST_YEARS_KEEP", "10"))

# =============================================================================
# Datasets
# =============================================================================
# Histórico principal (procesado offline)
HIST_DATASET = Path(
    os.getenv(
        "HIST_DATASET",
        str(PROCESSED_DIR / "dataset.parquet"),
    )
)

# LIVE principal (generado EXCLUSIVAMENTE por src/live_runner.py)
LIVE_DATASET = Path(
    os.getenv(
        "LIVE_DATASET",
        str(LIVE_DIR / "live_dataset.parquet"),
    )
)

# Metadata LIVE (trazabilidad técnica)
LIVE_META = Path(
    os.getenv(
        "LIVE_META",
        str(LIVE_DIR / "live_meta.json"),
    )
)

# LIVE alterno (opcional / legacy)
# Solo si usas ingest_arxiv.py directamente
LIVE_ARXIV = Path(
    os.getenv(
        "LIVE_ARXIV",
        str(PROCESSED_DIR / "live_arxiv.parquet"),
    )
)

# =============================================================================
# Runner LIVE
# =============================================================================
# Forma recomendada (evita errores de imports con src)
# Usado por sidebar.py
LIVE_RUNNER_MODULE = [sys.executable, "-m", "src.live_runner"]

# Fallback directo (NO recomendado, solo debug)
LIVE_RUNNER_SCRIPT = [
    sys.executable,
    str(PROJECT_ROOT / "src" / "live_runner.py"),
]

# Alias estable
LIVE_RUNNER = LIVE_RUNNER_MODULE

# =============================================================================
# Agregados macro-áreas (opcional)
# =============================================================================
AREA_TS = Path(
    os.getenv(
        "AREA_TS",
        str(PROCESSED_DIR / "area_ts.parquet"),
    )
)

AREA_TOP = Path(
    os.getenv(
        "AREA_TOP",
        str(PROCESSED_DIR / "area_top.parquet"),
    )
)

# =============================================================================
# Artefactos extra (opcional)
# =============================================================================
TRENDS_PARQUET = Path(
    os.getenv(
        "TRENDS_PARQUET",
        str(PROCESSED_DIR / "trends.parquet"),
    )
)

TRENDS_FORECAST = Path(
    os.getenv(
        "TRENDS_FORECAST",
        str(PROCESSED_DIR / "trends_forecast.parquet"),
    )
)

# =============================================================================
# Helpers
# =============================================================================
def ensure_dirs() -> None:
    """
    Asegura que la estructura mínima exista.
    Se llama al importar config.py (safe).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)


ensure_dirs()
