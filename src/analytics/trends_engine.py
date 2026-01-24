from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# =============================
# Configuración
# =============================
CACHE_TTL_SEC = 300
MAX_FEATURES = 3000
TOPK_CANDIDATES = 90

SPANISH_STOP_MINI = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no",
    "una","su","al","lo","como","mas","pero","sus","le","ya","o","este","si","porque",
    "esta","entre","cuando","muy","sin","sobre","tambien","me","hasta","hay","donde",
    "quien","desde","todo","nos","durante","todos","uno","les","ni","contra","otros",
}

DOMAIN_STOPWORDS = {
    "model","models","method","methods","approach","approaches","paper","work","results",
    "data","dataset","datasets","training","trained","task","tasks","performance","framework",
    "language","large","based","using","use","used","propose","proposed","introduce","introduced",
    "analysis","evaluation","experiment","experiments","show","demonstrate","demonstrates",
    "learning","neural","network","networks","system","systems","problem","problems",
    "state","art","existing","novel","new","present","provide","study",
}

STOPWORDS_ALL = set(ENGLISH_STOP_WORDS).union(SPANISH_STOP_MINI).union(DOMAIN_STOPWORDS)

# =============================
# Utilidades internas
# =============================
def _safe_period_series(df: pd.DataFrame, freq: str) -> pd.Series:
    return df["date"].dt.to_period(freq).astype(str)

# =============================
# Selección de términos
# =============================
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def pick_candidate_terms(df: pd.DataFrame, ngram_max: int, min_df: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["term", "score"])

    vec = TfidfVectorizer(
        stop_words=list(STOPWORDS_ALL),
        lowercase=True,
        ngram_range=(1, int(ngram_max)),
        max_features=MAX_FEATURES,
        min_df=max(1, int(min_df)),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
    )

    X = vec.fit_transform(df["text"].astype(str).values)
    terms = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1

    return (
        pd.DataFrame({"term": terms, "score": scores})
        .sort_values("score", ascending=False)
        .head(TOPK_CANDIDATES)
        .reset_index(drop=True)
    )

# =============================
# Series temporales por término
# =============================
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def build_term_matrix(df: pd.DataFrame, freq: str, terms: Tuple[str, ...]) -> pd.DataFrame:
    if df.empty or not terms:
        return pd.DataFrame()

    tmp = df[["date", "text"]].copy()
    tmp["period"] = _safe_period_series(tmp, freq)

    periods = sorted(tmp["period"].unique().tolist())
    if not periods:
        return pd.DataFrame()

    mat = pd.DataFrame(index=periods)
    text_series = tmp["text"]

    for t in terms:
        try:
            mask = text_series.str.contains(t, case=False, na=False, regex=False)
            counts = tmp.loc[mask].groupby("period").size()
            mat[t] = counts.reindex(periods, fill_value=0)
        except Exception:
            mat[t] = 0

    mat.index.name = "period"
    return mat

# =============================
# Clasificación de tendencias
# =============================
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def classify_from_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.empty or mat.shape[0] < 2:
        return pd.DataFrame()

    x = np.arange(mat.shape[0], dtype=float)
    vx = float(np.var(x))
    if vx <= 0:
        return pd.DataFrame()

    rows: List[Dict] = []

    for term in mat.columns:
        y = mat[term].astype(float).values
        total = int(y.sum())
        if total <= 0:
            continue

        slope = float(np.cov(x, y, bias=True)[0, 1] / vx)
        first = float(y[0])
        last = float(y[-1])
        growth = float((last - first) / (first + 1.0))
        mean = float(np.mean(y))
        std = float(np.std(y))
        stability = float(max(0.0, 1.0 - (std / (mean + 1.0))))

        label = "estable"
        if slope > 0.25 and growth > 0.30:
            label = "creciendo"
        elif slope < -0.25 and growth < -0.30:
            label = "bajando"

        rows.append({
            "term": term,
            "total": total,
            "slope": slope,
            "growth": growth,
            "stability": stability,
            "label": label,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    order = {"creciendo": 0, "estable": 1, "bajando": 2}
    out["ord"] = out["label"].map(order).fillna(9).astype(int)

    return (
        out.sort_values(["ord", "total"], ascending=[True, False])
        .drop(columns=["ord"])
        .reset_index(drop=True)
    )

# =============================
# Helpers públicos para la UI
# (IMPORTANTE: usados por widgets.py)
# =============================
def date_range_str(df: pd.DataFrame) -> str:
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
    if df is None or df.empty or "date" not in df.columns:
        return 0
    try:
        s = pd.to_datetime(df["date"], errors="coerce").dropna()
        if s.empty:
            return 0
        return int(s.dt.to_period(freq).nunique())
    except Exception:
        return 0
