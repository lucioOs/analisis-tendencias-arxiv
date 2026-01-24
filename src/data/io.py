# src/data/io.py
from __future__ import annotations

import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

from src.config import CACHE_TTL_SEC


def file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file()
    except Exception:
        return False


def human_time_ago(ts: float) -> str:
    delta = max(0, time.time() - ts)
    if delta < 60:
        return f"hace {int(delta)} s"
    if delta < 3600:
        return f"hace {int(delta // 60)} min"
    if delta < 86400:
        return f"hace {int(delta // 3600)} h"
    return f"hace {int(delta // 86400)} días"


def safe_last_update_label(p: Optional[Path], friendly_name: str) -> str:
    try:
        if not p or not file_exists(p):
            return f"{friendly_name}: no disponible"
        ts = p.stat().st_mtime
        return f"{friendly_name}: actualizado {human_time_ago(ts)}"
    except Exception:
        return f"{friendly_name}: no disponible"


def run_script_capture(args: List[str], timeout_sec: int = 900) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, "Se tardó demasiado y se canceló (timeout)."
    except Exception as e:
        return 1, f"Error ejecutando script: {e}"


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def read_parquet_cached(path_str: str, mtime: float) -> pd.DataFrame:
    # mtime se usa para invalidar cache automáticamente
    return pd.read_parquet(path_str)


def read_parquet_safe(path: Path, friendly_name: str) -> pd.DataFrame:
    try:
        if not file_exists(path):
            return pd.DataFrame()
        return read_parquet_cached(str(path), path.stat().st_mtime)
    except Exception as e:
        st.error(f"No se pudo leer {friendly_name}. Detalle: {e}")
        return pd.DataFrame()
