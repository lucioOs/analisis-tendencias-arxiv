from __future__ import annotations

import pandas as pd


def date_range_str(df: pd.DataFrame) -> str:
    """Devuelve 'YYYY-MM-DD → YYYY-MM-DD' usando columna date."""
    if df is None or df.empty or "date" not in df.columns:
        return "Sin datos"
    try:
        s = pd.to_datetime(df["date"], errors="coerce").dropna()
        if s.empty:
            return "Sin datos"
        return f"{s.min().date()} → {s.max().date()}"
    except Exception:
        return "Sin datos"


def period_count(df: pd.DataFrame, freq: str) -> int:
    """Cuenta periodos únicos según freq ('D', 'W', 'M')."""
    if df is None or df.empty or "date" not in df.columns:
        return 0
    try:
        s = pd.to_datetime(df["date"], errors="coerce").dropna()
        if s.empty:
            return 0
        return int(s.dt.to_period(freq).nunique())
    except Exception:
        return 0


def limit_df(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Limita el DF a los últimos max_rows (ordenado por date)."""
    if df is None or df.empty:
        return df
    max_rows = int(max_rows)
    if len(df) <= max_rows:
        return df
    if "date" not in df.columns:
        return df.tail(max_rows).copy()
    return df.sort_values("date").tail(max_rows).copy()
