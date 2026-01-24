from __future__ import annotations

import re
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.taxonomy import MACRO_AREAS  # type: ignore
    TAXONOMY_OK = True
except Exception:
    MACRO_AREAS = {}
    TAXONOMY_OK = False


def taxonomy_available() -> bool:
    return bool(TAXONOMY_OK and MACRO_AREAS)


def _compile_area_patterns(macro_areas: dict) -> Dict[str, re.Pattern]:
    pats: Dict[str, re.Pattern] = {}
    for area, kws in (macro_areas or {}).items():
        escaped = [re.escape(k.strip().lower()) for k in kws if isinstance(k, str) and k.strip()]
        if not escaped:
            continue
        pat = re.compile(r"(" + "|".join(escaped) + r")", flags=re.IGNORECASE)
        pats[area] = pat
    return pats


@st.cache_data(show_spinner=False, ttl=300)
def compute_area_timeseries_from_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty or not taxonomy_available():
        return pd.DataFrame()

    pats = _compile_area_patterns(MACRO_AREAS)

    tmp = df[["date", "text"]].copy()
    tmp["period"] = tmp["date"].dt.to_period(freq).dt.start_time

    rows = []
    for area, pat in pats.items():
        m = tmp["text"].astype(str).str.lower().str.contains(pat, na=False)
        part = tmp.loc[m, ["period"]].copy()
        if part.empty:
            continue
        g = part.groupby("period").size().rename("count_docs").reset_index()
        g["area"] = area
        rows.append(g)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    totals = tmp.groupby("period").size().rename("total_docs").reset_index()
    out = out.merge(totals, on="period", how="left")
    out["rel_docs"] = (out["count_docs"] / out["total_docs"].replace(0, np.nan)).fillna(0.0)
    out = out.sort_values(["area", "period"]).reset_index(drop=True)
    return out
