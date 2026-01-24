# src/data/normalize.py
from __future__ import annotations

from typing import Set
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.config import CACHE_TTL_SEC

SPANISH_STOP_MINI: Set[str] = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no",
    "una","su","al","lo","como","mas","pero","sus","le","ya","o","este","si","porque",
    "esta","entre","cuando","muy","sin","sobre","tambien","me","hasta","hay","donde",
    "quien","desde","todo","nos","durante","todos","uno","les","ni","contra","otros",
}

DOMAIN_STOPWORDS: Set[str] = {
    "model","models","method","methods","approach","approaches","paper","work","results",
    "data","dataset","datasets","training","trained","task","tasks","performance","framework",
    "language","large","based","using","use","used","propose","proposed","introduce","introduced",
    "analysis","evaluation","experiment","experiments","show","demonstrate","demonstrates",
    "learning","neural","network","networks","system","systems","problem","problems",
    "state","art","existing","novel","new","present","provide","study",
}

STOPWORDS_ALL: Set[str] = set(ENGLISH_STOP_WORDS).union(SPANISH_STOP_MINI).union(DOMAIN_STOPWORDS)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def normalize_base_cached(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "text", "source"])

    cols = {c.lower().strip(): c for c in df.columns}

    date_col = None
    for k in [
        "date","published","published_at","updated","created",
        "timestamp","time","submitteddate","submitted_date","update_date","published_date",
    ]:
        if k in cols:
            date_col = cols[k]
            break

    text_col = None
    for k in [
        "text","summary","abstract","content","title","paper_title","paper",
        "description","descripcion","body",
    ]:
        if k in cols:
            text_col = cols[k]
            break

    title_col = cols.get("title", None)
    abstract_col = cols.get("abstract", None)
    source_col = cols.get("source", None)

    out = pd.DataFrame()

    if date_col is not None:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        y = pd.to_numeric(df[cols["year"]], errors="coerce") if "year" in cols else None
        m = pd.to_numeric(df[cols["month"]], errors="coerce") if "month" in cols else None
        d = pd.to_numeric(df[cols["day"]], errors="coerce") if "day" in cols else None
        if y is not None and m is not None:
            out["date"] = pd.to_datetime(
                dict(
                    year=y.fillna(2000).astype(int),
                    month=m.fillna(1).astype(int),
                    day=(d.fillna(1).astype(int) if d is not None else 1),
                ),
                errors="coerce",
                utc=True,
            ).dt.tz_convert(None)
        else:
            return pd.DataFrame(columns=["date", "text", "source"])

    if text_col is not None:
        out["text"] = df[text_col].astype(str)
    elif title_col is not None and abstract_col is not None:
        out["text"] = df[title_col].astype(str) + ". " + df[abstract_col].astype(str)
    else:
        return pd.DataFrame(columns=["date", "text", "source"])

    if source_col is not None:
        out["source"] = df[source_col].astype(str)
    else:
        out["source"] = "dataset"

    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 20]
    out = out.sort_values("date").reset_index(drop=True)

    try:
        out["source"] = out["source"].astype("category")
    except Exception:
        pass

    return out[["date", "text", "source"]]


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return normalize_base_cached(df)
    except Exception:
        return pd.DataFrame(columns=["date", "text", "source"])
