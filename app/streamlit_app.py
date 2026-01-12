# app/streamlit_app.py
# Dashboard de Tendencias (PLN + ML) - Versión final robusta (con optimizaciones de memoria)
# Incluye:
# - Histórico (MIT): tendencias por clase + serie temporal
# - Live (arXiv/RSS): actualización + fallback (actividad) + clasificación heurística
# - N-grams, WordCloud, KPIs, tablas
# - Optimización: cierre de figuras, caching con TTL, límites de datos para evitar consumo de RAM

from __future__ import annotations

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


# -----------------------------
# Config / Paths
# -----------------------------
ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"

HIST_DATASET = PROCESSED / "dataset.parquet"
LIVE_DATASET = PROCESSED / "live_dataset.parquet"

TRENDS_FULL = PROCESSED / "trends_full.parquet"
TREND_CLASSES = PROCESSED / "trend_classes.parquet"

EXPORT_LIVE = ROOT / "src" / "export_live_dataset.py"

APP_TITLE = "Sistema de Análisis y Predicción de Tendencias (PLN + ML)"


# -----------------------------
# Parámetros de performance / memoria (ajustables)
# -----------------------------
CACHE_TTL_SEC = 300              # 5 min: evita cache indefinido en sesiones largas
MAX_LIVE_ROWS = 3000             # límite de filas usadas en cálculos Live
MAX_WC_DOCS = 250                # máximo de documentos para WordCloud
MAX_RECENT_ROWS = 40             # tabla de recientes
TOPK_ACTIVITY_TERMS = 40         # top términos en actividad
TOPK_CANDIDATES = 60             # candidatos TF-IDF para series/heurística
MAX_FEATURES_ACTIVITY = 4000     # vocabulario TF-IDF para actividad
MAX_FEATURES_TRENDS = 1500       # vocabulario TF-IDF para tendencias


# -----------------------------
# Utilidades (safe I/O, procesos)
# -----------------------------
def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def read_parquet_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not file_exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        return read_parquet_cached(str(path))
    except Exception as e:
        st.error(f"Error al leer {path.name}: {e}")
        return pd.DataFrame()


def run_script_capture(args: List[str], timeout_sec: int = 240) -> Tuple[int, str]:
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
        return 124, "Tiempo de espera excedido."
    except Exception as e:
        return 1, f"Excepción: {e}"


# -----------------------------
# Normalización de datos
# -----------------------------
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def normalize_base_cached(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}

    date_col = next((cols[k] for k in ["date", "published", "published_at", "created", "published date"] if k in cols), None)
    text_col = next((cols[k] for k in ["text", "content", "summary", "abstract", "title", "article body", "article header"] if k in cols), None)
    source_col = next((cols[k] for k in ["source", "origin"] if k in cols), None)

    if date_col is None or text_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    # Parse tolerante de fechas; utc=True solo si vienes de feeds con tz
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    out["text"] = df[text_col].astype(str)

    if source_col:
        out["source"] = df[source_col].astype(str)
    else:
        out["source"] = "unknown"

    # reduce memoria en source
    out["source"] = out["source"].astype("category")

    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.dropna(subset=["date", "text"])
    out = out[out["text"].str.len() >= 20]
    out = out.sort_values("date").reset_index(drop=True)
    return out


def normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return normalize_base_cached(df)
    except Exception:
        # fallback sin cache si streamlit no puede hashear el DF en algún entorno
        return normalize_base_cached.clear() or pd.DataFrame()


def period_count(df: pd.DataFrame, freq: str) -> int:
    if df.empty:
        return 0
    try:
        return df["date"].dt.to_period(freq).nunique()
    except Exception:
        return 0


def date_range_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "-"
    try:
        return f"{df['date'].min().date()} -> {df['date'].max().date()}"
    except Exception:
        return "-"


def limit_live_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # limitar para estabilidad si la app corre horas
    df = df.sort_values("date").tail(MAX_LIVE_ROWS).copy()
    return df


# -----------------------------
# WordCloud (optimizado)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def join_text_for_wc(texts: List[str]) -> str:
    # Cachea la concatenación para evitar recomputar en reruns
    return " ".join(texts)


def plot_wordcloud_from_text(text: str) -> Optional[plt.Figure]:
    if not text or len(text.strip()) < 50:
        return None
    try:
        wc = WordCloud(
            width=900,
            height=380,
            background_color=None,
            mode="RGBA",
            colormap="viridis",
            stopwords=ENGLISH_STOP_WORDS,
            min_word_length=4,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_alpha(0)
        return fig
    except Exception:
        return None


# -----------------------------
# Actividad Live (fallback)
# -----------------------------
@dataclass
class LiveActivityResult:
    top_terms: pd.DataFrame
    activity_by_day: pd.DataFrame
    recent_rows: pd.DataFrame


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def compute_live_activity_cached(dfw: pd.DataFrame, ngram_max: int, min_df: int) -> LiveActivityResult:
    if dfw.empty:
        return LiveActivityResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES_ACTIVITY,
        ngram_range=(1, max(1, int(ngram_max))),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]+\b",
        min_df=max(1, int(min_df)),
    )

    try:
        X = vectorizer.fit_transform(dfw["text"].values)
        terms = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1

        idx = scores.argsort()[::-1][:TOPK_ACTIVITY_TERMS]
        top_terms = pd.DataFrame({"term": terms[idx], "score": scores[idx]})
    except Exception:
        top_terms = pd.DataFrame(columns=["term", "score"])

    tmp = dfw.copy()
    tmp["day"] = tmp["date"].dt.date
    activity = tmp.groupby("day", as_index=False).size().rename(columns={"size": "items"})

    recent = dfw.sort_values("date", ascending=False).head(MAX_RECENT_ROWS)[["date", "source", "text"]]
    recent["text"] = recent["text"].str.slice(0, 350)

    return LiveActivityResult(top_terms, activity, recent)


def compute_live_activity(dfw: pd.DataFrame, ngram_max: int, min_df: int) -> LiveActivityResult:
    return compute_live_activity_cached(dfw, int(ngram_max), int(min_df))


# -----------------------------
# Clasificación heurística Live
# -----------------------------
@dataclass
class TrendRow:
    term: str
    slope: float
    growth: float
    stability: float
    total: int
    label: str


def _safe_period_series(dfw: pd.DataFrame, freq: str) -> pd.Series:
    return dfw["date"].dt.to_period(freq).astype(str)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def build_term_timeseries_cached(dfw: pd.DataFrame, freq: str, terms: Tuple[str, ...]) -> pd.DataFrame:
    if dfw.empty or not terms:
        return pd.DataFrame()

    tmp = dfw[["date", "text"]].copy()
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


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SEC)
def classify_trends_from_matrix_cached(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.empty or mat.shape[0] < 2:
        return pd.DataFrame()

    x = np.arange(mat.shape[0], dtype=float)
    vx = float(np.var(x))
    if vx <= 0:
        return pd.DataFrame()

    rows: list[TrendRow] = []

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

        label = "otro"
        if slope > 0.25 and growth > 0.30:
            label = "emergente"
        elif slope < -0.25 and growth < -0.30:
            label = "declive"
        else:
            label = "candidato_consolidada"

        rows.append(TrendRow(term, slope, growth, stability, total, label))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame([r.__dict__ for r in rows])

    p70 = float(out["total"].quantile(0.70))
    mask_cons = (
        (out["label"] == "candidato_consolidada")
        & (out["total"] >= p70)
        & (out["stability"] >= 0.55)
        & (out["slope"].abs() < 0.25)
    )
    out.loc[mask_cons, "label"] = "consolidada"
    out.loc[out["label"] == "candidato_consolidada", "label"] = "otro"

    out = out.sort_values(["label", "slope", "total"], ascending=[True, False, False]).reset_index(drop=True)
    return out


# -----------------------------
# UI helpers
# -----------------------------
def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #464b59;
            padding: 10px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_kpis(df: pd.DataFrame, context_label: str, freq: str):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total documentos", f"{len(df):,}")
    c2.metric("Cobertura temporal", date_range_str(df))
    c3.metric(f"Periodos ({freq})", f"{period_count(df, freq)}")


# -----------------------------
# Vistas
# -----------------------------
def historic_view():
    st.subheader("Análisis histórico (MIT)")
    st.caption("Análisis longitudinal para entrenamiento y validación de modelos.")

    df_base = normalize_base(read_parquet_safe(HIST_DATASET))
    if df_base.empty:
        st.error("No se encontró la base histórica procesada (dataset.parquet).")
        return

    show_kpis(df_base, "Histórico", "M")

    df_trends = read_parquet_safe(TRENDS_FULL)
    df_classes = read_parquet_safe(TREND_CLASSES)

    if df_trends.empty or df_classes.empty:
        st.warning("Faltan archivos de tendencias (trends_full.parquet / trend_classes.parquet).")
        st.dataframe(df_base.head(50), use_container_width=True)
        return

    df = df_trends.merge(df_classes, on="term", how="left")
    df["class"] = df["class"].fillna("sin_clase")

    st.divider()
    classes_order = ["emergente", "consolidada", "declive"]
    tabs = st.tabs(["Emergente", "Consolidada", "Declive"])

    for cls, tab in zip(classes_order, tabs):
        with tab:
            sub = df[df["class"] == cls].copy()
            if sub.empty:
                st.info(f"No hay tendencias clasificadas como {cls}.")
                continue

            top_terms = (
                sub.groupby("term")["count"].sum()
                .sort_values(ascending=False)
                .head(150)
                .index.tolist()
            )

            col_sel, col_plot = st.columns([1, 2])
            with col_sel:
                selected_term = st.selectbox(f"Selecciona tendencia ({cls})", top_terms, key=f"hist_sel_{cls}")

            ts = sub[sub["term"] == selected_term].sort_values("period")
            with col_plot:
                if "rel_freq" in ts.columns:
                    st.line_chart(ts.set_index("period")["rel_freq"], use_container_width=True)
                else:
                    st.line_chart(ts.set_index("period")["count"], use_container_width=True)

            with st.expander("Ver datos"):
                st.dataframe(ts, use_container_width=True)


def live_view():
    st.subheader("Monitor Live (arXiv + RSS)")
    st.caption("Modo Live. Si la historia es corta, se muestra actividad general.")

    with st.sidebar:
        st.header("Configuración Live")
        window_days = st.slider("Ventana (días)", 7, 365, 180)
        freq = st.selectbox("Frecuencia", ["D", "W", "M"], index=1)

        st.divider()
        st.subheader("Parámetros NLP")
        ngram_max = st.selectbox("N-grams", [1, 2, 3], index=1)
        min_df = st.number_input("min_df", 1, 10, 2)

        if st.button("Actualizar ahora", type="primary"):
            if not file_exists(EXPORT_LIVE):
                st.error("No se encontró src/export_live_dataset.py")
            else:
                with st.spinner("Actualizando Live..."):
                    code, out = run_script_capture([sys.executable, str(EXPORT_LIVE)], timeout_sec=300)
                    if code == 0:
                        st.success("Live actualizado.")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Falló la actualización Live.")
                        st.text(out[:4000])

    df_live = normalize_base(read_parquet_safe(LIVE_DATASET))
    if df_live.empty:
        st.warning("No hay datos Live. Presiona 'Actualizar ahora'.")
        return

    cutoff = df_live["date"].max() - pd.Timedelta(days=int(window_days))
    dfw = df_live[df_live["date"] >= cutoff].copy()

    # Fallback si la ventana deja el conjunto vacío
    if dfw.empty:
        dfw = df_live.tail(300).copy()
        st.info("La ventana seleccionada dejó el conjunto vacío. Se muestran los últimos 300 registros.")

    dfw = limit_live_df(dfw)

    show_kpis(dfw, "Live", freq)

    with st.expander("Nube de conceptos", expanded=True):
        texts_wc = dfw.sort_values("date", ascending=False)["text"].head(MAX_WC_DOCS).astype(str).tolist()
        wc_text = join_text_for_wc(texts_wc)
        fig = plot_wordcloud_from_text(wc_text)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)  # libera memoria
        else:
            st.info("No hay datos suficientes para generar nube de palabras.")

    n_periods = period_count(dfw, freq)

    # Fallback: actividad
    if n_periods < 2:
        st.info(
            f"Modo actividad. Periodos detectados: {n_periods}. "
            "Se requieren al menos 2 periodos para calcular tendencias."
        )

        res = compute_live_activity(dfw, ngram_max, min_df)
        left, right = st.columns(2)

        with left:
            st.subheader("Términos más activos")
            if res.top_terms.empty:
                st.info("No hay términos detectables. Baja min_df o cambia ngram_max.")
            else:
                st.dataframe(
                    res.top_terms,
                    use_container_width=True,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Relevancia (TF-IDF)",
                            format="%.2f",
                            min_value=0,
                            max_value=float(res.top_terms["score"].max()),
                        )
                    },
                )

        with right:
            st.subheader("Actividad por día")
            if res.activity_by_day.empty:
                st.info("No hay actividad agregable.")
            else:
                st.bar_chart(res.activity_by_day.set_index("day"), use_container_width=True)

        st.subheader("Documentos recientes")
        st.dataframe(res.recent_rows, use_container_width=True)
        return

    # Tendencias: clasificación heurística
    st.success(f"Modo tendencias. Historia suficiente: {n_periods} periodos.")

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, int(ngram_max)),
        max_features=MAX_FEATURES_TRENDS,
        min_df=int(min_df),
    )

    try:
        X = vec.fit_transform(dfw["text"])
        scores = X.sum(axis=0).A1
        terms = vec.get_feature_names_out()

        cand = (
            pd.DataFrame({"term": terms, "score": scores})
            .sort_values("score", ascending=False)
            .head(TOPK_CANDIDATES)
            .reset_index(drop=True)
        )

        if cand.empty:
            st.info("No se detectaron candidatos. Ajusta min_df o ngram_max.")
            return

        mat = build_term_timeseries_cached(dfw, freq, tuple(cand["term"].tolist()))
        if mat.empty or mat.shape[0] < 2:
            st.info("No hay suficientes periodos para construir series temporales.")
            return

        cls = classify_trends_from_matrix_cached(mat)
        if cls.empty:
            st.info("No se pudo clasificar ningún término.")
            return

        st.subheader("Clasificación Live")
        tabs = st.tabs(["Emergente", "Consolidada", "Declive", "Otros"])

        def render_class(tab, label_name: str, sort_mode: str):
            with tab:
                sub = cls[cls["label"] == label_name].copy()
                if sub.empty:
                    st.info(f"No hay elementos en {label_name}.")
                    return

                if sort_mode == "slope":
                    sub = sub.sort_values(["slope", "total"], ascending=[False, False])
                else:
                    sub = sub.sort_values(["total", "stability"], ascending=[False, False])

                st.dataframe(
                    sub[["term", "total", "slope", "growth", "stability"]],
                    use_container_width=True,
                    column_config={
                        "term": "Término",
                        "total": "Total",
                        "slope": st.column_config.NumberColumn("Pendiente", format="%.3f"),
                        "growth": st.column_config.NumberColumn("Crecimiento", format="%.2f"),
                        "stability": st.column_config.NumberColumn("Estabilidad", format="%.2f"),
                    },
                )

                top_terms = sub["term"].head(50).tolist()
                selected = st.selectbox("Selecciona un término", top_terms, key=f"live_sel_{label_name}")

                series = mat[selected].reset_index().rename(columns={selected: "items"})
                st.line_chart(series.set_index("period")["items"], use_container_width=True)

        render_class(tabs[0], "emergente", "slope")
        render_class(tabs[1], "consolidada", "total")
        render_class(tabs[2], "declive", "slope")

        with tabs[3]:
            sub = cls[cls["label"] == "otro"].copy()
            if sub.empty:
                st.info("No hay elementos en otros.")
            else:
                st.dataframe(sub[["term", "total", "slope", "growth", "stability"]].head(80), use_container_width=True)

        with st.expander("Criterios de clasificación"):
            st.write(
                "Se generan candidatos por TF-IDF y se construye una serie temporal (conteos por periodo). "
                "Pendiente: regresión lineal sobre conteos. Crecimiento: último vs primero. "
                "Estabilidad: variación normalizada. "
                "Emergente: pendiente alta y crecimiento positivo. Declive: pendiente negativa y crecimiento negativo. "
                "Consolidada: alta presencia, estable y pendiente cercana a cero."
            )

    except Exception as e:
        st.error(f"Error en clasificación Live: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_custom_css()

    st.title(APP_TITLE)

    mode = st.radio("Modo", ["Histórico (MIT)", "Live (Tiempo real)"], horizontal=True)

    if "Live" in mode:
        live_view()
    else:
        historic_view()


if __name__ == "__main__":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    main()
