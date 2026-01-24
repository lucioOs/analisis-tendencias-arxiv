from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analytics.trends_engine import STOPWORDS_ALL

CACHE_TTL_SEC = 300
MAX_WC_DOCS = 600
MAX_WC_STRATA = 24

try:
    from wordcloud import WordCloud  # type: ignore
    WORDCLOUD_OK = True
except Exception:
    WordCloud = None  # type: ignore
    WORDCLOUD_OK = False


def _stratified_time_sample(df: pd.DataFrame, max_docs: int, strata: int = MAX_WC_STRATA, seed: int = 7) -> List[str]:
    if df.empty:
        return []
    df = df.sort_values("date").copy()
    n = len(df)
    if n <= max_docs:
        return df["text"].astype(str).tolist()

    strata = max(4, int(strata))
    per = max(1, int(max_docs // strata))
    out_texts: List[str] = []

    edges = np.linspace(0, n, strata + 1).astype(int)
    rng = np.random.default_rng(seed)
    for i in range(strata):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            continue
        part = df.iloc[a:b]
        if len(part) <= per:
            out_texts.extend(part["text"].astype(str).tolist())
        else:
            idx = rng.choice(len(part), size=per, replace=False)
            out_texts.extend(part.iloc[idx]["text"].astype(str).tolist())

    return out_texts[:max_docs]


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def _wc_freq_from_tfidf(texts: Tuple[str, ...]) -> Dict[str, float]:
    if not texts:
        return {}

    vec = TfidfVectorizer(
        stop_words=list(STOPWORDS_ALL),
        lowercase=True,
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
    )
    X = vec.fit_transform(list(texts))
    terms = vec.get_feature_names_out()
    weights = np.asarray(X.sum(axis=0)).ravel().astype(float)

    freq: Dict[str, float] = {}
    for t, w in zip(terms, weights):
        if w > 0 and len(t) >= 3:
            freq[t] = float(w)
    return freq


def _plot_from_freq(freq: Dict[str, float]) -> Optional[plt.Figure]:
    if not WORDCLOUD_OK or not freq:
        return None
    try:
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=None,
            mode="RGBA",
            collocations=False,
            max_words=250,
        ).generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0)
        return fig
    except Exception:
        return None


def _plot_from_text(text: str) -> Optional[plt.Figure]:
    if not WORDCLOUD_OK or not text or len(text.strip()) < 80:
        return None
    try:
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=None,
            mode="RGBA",
            stopwords=STOPWORDS_ALL,
            min_word_length=3,
            collocations=False,
            max_words=250,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0)
        return fig
    except Exception:
        return None


def render_wordcloud(df: pd.DataFrame, title: str, mode: str):
    with st.expander(title, expanded=True):
        if df.empty:
            st.info("No hay datos para generar la nube.")
            return
        if not WORDCLOUD_OK:
            st.info("WordCloud no está disponible en este entorno.")
            return

        texts = _stratified_time_sample(df, max_docs=MAX_WC_DOCS, strata=MAX_WC_STRATA)

        if mode == "Destacados (TF-IDF)":
            freq = _wc_freq_from_tfidf(tuple(texts))
            fig = _plot_from_freq(freq)
        else:
            fig = _plot_from_text(" ".join(texts))

        if fig is None:
            st.info("No se pudo generar la nube (poca información o formato irregular).")
        else:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
