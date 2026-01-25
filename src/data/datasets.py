# src/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from src.config import HIST_DATASET, HIST_YEARS_KEEP, LIVE_DATASET, LIVE_META, MAX_ROWS_TEXT
from src.data.io import file_exists

CACHE_TTL_SEC = 300


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    label: str


def _read_parquet_columns(path: Path, cols: list[str]) -> pd.DataFrame:
    """
    Intenta lectura por columnas (más rápido).
    Si el engine/archivo no soporta columns=, cae a lectura completa.
    """
    try:
        return pd.read_parquet(str(path), columns=cols)
    except Exception:
        try:
            return pd.read_parquet(str(path))
        except Exception:
            return pd.DataFrame()


def _ensure_min_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura un schema mínimo para la UI:
    - date: datetime
    - text: string (si no existe, se arma con title+abstract)
    - categories, id: string
    """
    if df.empty:
        return df

    df2 = df.copy()

    # date
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce", utc=False)
    else:
        df2["date"] = pd.NaT

    # text
    if "text" not in df2.columns:
        if "title" in df2.columns and "abstract" in df2.columns:
            df2["text"] = (df2["title"].astype(str) + " " + df2["abstract"].astype(str)).astype(str)
        elif "title" in df2.columns:
            df2["text"] = df2["title"].astype(str)
        else:
            df2["text"] = ""

    # categories / id opcionales
    if "categories" not in df2.columns:
        df2["categories"] = ""
    if "id" not in df2.columns:
        df2["id"] = ""

    df2 = df2.dropna(subset=["date"]).copy()
    df2["text"] = df2["text"].astype(str)
    df2["categories"] = df2["categories"].astype(str)
    df2["id"] = df2["id"].astype(str)

    # title/abstract opcionales para UI (si existen, garantizamos str)
    if "title" in df2.columns:
        df2["title"] = df2["title"].astype(str)
    if "abstract" in df2.columns:
        df2["abstract"] = df2["abstract"].astype(str)

    return df2


def _keep_last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Conserva últimos N años respecto al max(date)."""
    if df.empty or years <= 0:
        return df
    mx = df["date"].max()
    if pd.isna(mx):
        return df
    cut = mx - pd.Timedelta(days=int(years) * 365)
    return df[df["date"] >= cut].copy()


def _thin_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evita que la UI se muera:
    - si hay demasiado, toma un muestreo determinístico (estable)
    """
    if df.empty:
        return df
    n = len(df)
    if n <= MAX_ROWS_TEXT:
        return df

    df2 = df.sort_values("date")
    step = max(1, n // MAX_ROWS_TEXT)
    return df2.iloc[::step].copy()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def _load_cached(mtime: float, path_str: str, mode: str) -> pd.DataFrame:
    """
    Cache por:
    - path_str
    - mtime (invalida cache cuando cambia el archivo)
    """
    path = Path(path_str)

    # lectura rápida por columnas (si se puede)
    df = _read_parquet_columns(path, cols=["date", "text", "categories", "id", "title", "abstract"])

    df = _ensure_min_schema(df)

    # histórico: últimos N años
    if mode == "hist":
        df = _keep_last_years(df, HIST_YEARS_KEEP)

    # UI: adelgazar dataset
    df = _thin_for_ui(df)

    return df


def _load_dataset(path: Path, mode: str) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()
    try:
        return _load_cached(path.stat().st_mtime, str(path), mode)
    except Exception:
        return pd.DataFrame()


def load_historico_dataset() -> Tuple[pd.DataFrame, str, Path]:
    p = Path(HIST_DATASET)
    df = _load_dataset(p, mode="hist")
    if df.empty:
        return df, "Histórico (sin datos)", p

    mx = df["date"].max()
    mn = df["date"].min()
    label = f"Histórico (últimos {HIST_YEARS_KEEP} años · {mn.date()} → {mx.date()})"
    return df, label, p


def _live_label_from_meta() -> str:
    """
    Construye un label corto desde data/live/live_meta.json (si existe).
    """
    mp = Path(LIVE_META)
    if not file_exists(mp):
        return "Live"

    try:
        import json

        meta = json.loads(mp.read_text(encoding="utf-8"))
        src_used = meta.get("source_used") or meta.get("source") or None
        updated = meta.get("updated_at_utc") or None
        rows = meta.get("rows") or meta.get("total_rows") or None

        parts = ["Live"]
        if src_used:
            parts.append(f"fuente={src_used}")
        if rows is not None:
            parts.append(f"rows={rows}")
        if updated:
            parts.append(f"updated_at={updated}")

        return " (" + " · ".join(parts[1:]) + ")" if len(parts) > 1 else "Live"
    except Exception:
        return "Live"


def load_live_dataset() -> Tuple[pd.DataFrame, str, Path]:
    """
    Carga dataset LIVE (generado por src/live_runner.py).
    Ruta canónica en src/config.py:
      data/live/live_dataset.parquet
    """
    p = Path(LIVE_DATASET)
    df = _load_dataset(p, mode="live")
    if df.empty:
        # si el parquet no existe o está vacío, informamos de la ruta
        return df, f"Live (sin datos) · esperado: {p}", p

    mx = df["date"].max()
    mn = df["date"].min()

    meta_label = _live_label_from_meta()
    # meta_label puede ser "Live" o "(fuente=... · ...)" según meta
    if meta_label.startswith(" ("):
        label = f"Live{meta_label} · {mn} → {mx}"
    else:
        label = f"Live ({mn} → {mx})"

    return df, label, p
