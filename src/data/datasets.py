# src/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from src.config import HIST_DATASET, LIVE_DATASET, HIST_YEARS_KEEP, MAX_ROWS_TEXT
from src.data.io import file_exists


CACHE_TTL_SEC = 300


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    label: str


def _read_parquet_columns(path: Path, cols: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(str(path), columns=cols)
    except Exception:
        # fallback: intenta leer completo si el parquet no permite columns=
        try:
            return pd.read_parquet(str(path))
        except Exception:
            return pd.DataFrame()


def _ensure_min_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df.copy()

    # date
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
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

    # categories/id opcionales
    if "categories" not in df2.columns:
        df2["categories"] = ""
    if "id" not in df2.columns:
        df2["id"] = ""

    df2 = df2.dropna(subset=["date"]).copy()
    df2["text"] = df2["text"].astype(str)
    df2["categories"] = df2["categories"].astype(str)
    df2["id"] = df2["id"].astype(str)

    return df2


def _keep_last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
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
    - si hay demasiado, toma una muestra (pero estable) para análisis interactivo
    """
    if df.empty:
        return df
    n = len(df)
    if n <= MAX_ROWS_TEXT:
        return df

    # sample estable: ordena por date y toma espaciado (determinístico)
    df2 = df.sort_values("date")
    step = max(1, n // MAX_ROWS_TEXT)
    return df2.iloc[::step].copy()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def _load_cached(mtime: float, path_str: str, mode: str) -> pd.DataFrame:
    path = Path(path_str)

    # lee mínimo (rápido)
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
    return _load_cached(path.stat().st_mtime, str(path), mode)


def load_historico_dataset() -> Tuple[pd.DataFrame, str, Path]:
    p = Path(HIST_DATASET)
    df = _load_dataset(p, mode="hist")
    if df.empty:
        return df, "Histórico (sin datos)", p

    mx = df["date"].max()
    mn = df["date"].min()
    label = f"Histórico (últimos {HIST_YEARS_KEEP} años · {mn.date()} → {mx.date()})"
    return df, label, p


def load_live_dataset() -> Tuple[pd.DataFrame, str, Path]:
    p = Path(LIVE_DATASET)
    df = _load_dataset(p, mode="live")
    if df.empty:
        return df, "Live (sin datos)", p

    mx = df["date"].max()
    mn = df["date"].min()
    label = f"Live ({mn} → {mx})"
    return df, label, p
