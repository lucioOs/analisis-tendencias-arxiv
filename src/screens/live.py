# src/screens/live.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import MAX_ROWS_TEXT
from src.data.datasets import load_live_dataset
from src.data.io import safe_last_update_label
from src.metrics import limit_df, period_count
from src.ui.widgets import show_kpis, render_actions_header, render_recent, download_table
from src.ui.wordcloud_ui import render_wordcloud
from src.analytics.trends_engine import pick_candidate_terms, build_term_matrix, classify_from_matrix
from src.plotting.charts import plot_trend_periods, plot_forecast, tick_step
from src.taxonomy import MACRO_AREAS


# -----------------------------
# Helpers: Macro-área (PLN keywords) - LIVE
# -----------------------------
@st.cache_data(show_spinner=False, ttl=300)
def _compile_area_pattern(area: str) -> str:
    """
    Compila un patrón regex (string) para la macro-área.
    Cacheado para no recompilar continuamente.
    """
    if not area or area == "Todas":
        return ""
    kws = MACRO_AREAS.get(area, [])
    kws = [str(k).strip() for k in kws if str(k).strip()]
    if not kws:
        return ""
    return "|".join([re.escape(k) for k in kws])


def _filter_by_macro_area(df: pd.DataFrame, area: str) -> pd.DataFrame:
    """
    Filtra por macro-área usando keywords (PLN básico) sobre df['text'].
    """
    if df.empty or not area or area == "Todas":
        return df
    if "text" not in df.columns:
        return df

    pat = _compile_area_pattern(area)
    if not pat:
        return df

    mask = df["text"].astype(str).str.contains(pat, case=False, na=False, regex=True)
    return df[mask].copy()


# -----------------------------
# Helpers: LIVE window + tabla artículos
# -----------------------------
def _apply_live_window(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    """Conserva solo los últimos N días respecto al max(date). Si days es None/0, no filtra."""
    if df.empty:
        return df
    if not days or int(days) <= 0:
        return df
    if "date" not in df.columns:
        return df

    mx = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(mx):
        return df

    cut = mx - pd.Timedelta(days=int(days))
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date"])
    return dfx[dfx["date"] >= cut].copy()


def _filter_articles_by_term(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """Filtra artículos donde 'term' aparece (búsqueda simple) en title/abstract/text."""
    if df.empty or not term:
        return pd.DataFrame()

    term = term.strip().lower()
    parts: list[pd.Series] = []

    for c in ["title", "abstract", "text"]:
        if c in df.columns:
            parts.append(df[c].astype(str))

    if not parts:
        return pd.DataFrame()

    blob = parts[0]
    for p in parts[1:]:
        blob = blob + " " + p

    mask = blob.str.lower().str.contains(term, na=False)
    return df[mask].copy()


def _build_article_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara tabla de artículos para UI:
    date, title, abstract_short, categories, link, id
    """
    if df.empty:
        return df

    out = df.copy()

    # Fecha
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Título
    if "title" not in out.columns:
        out["title"] = ""

    # Abstract
    if "abstract" not in out.columns:
        out["abstract"] = out["text"].astype(str) if "text" in out.columns else ""

    # Link / id
    if "link" not in out.columns:
        out["link"] = ""
    if "id" not in out.columns:
        out["id"] = out["link"].astype(str)

    # Categorías
    if "categories" not in out.columns:
        out["categories"] = ""

    # Abstract truncado
    out["abstract_short"] = (
        out["abstract"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 350)
    )

    cols = ["date", "title", "abstract_short", "categories", "link", "id"]
    cols = [c for c in cols if c in out.columns]

    out = out[cols].sort_values("date", ascending=False)
    return out


def _render_articles_section(df_base: pd.DataFrame, term: str, prefix: str = "live") -> None:
    """UI: muestra tabla de artículos relacionados al término."""
    st.subheader("Artículos relacionados (muestra)")

    hits = _filter_articles_by_term(df_base, term)
    if hits.empty:
        st.info("No se encontraron artículos que contengan ese término en title/abstract/text.")
        return

    max_articles = st.slider(
        "Máximo de artículos a mostrar",
        min_value=20,
        max_value=300,
        value=80,
        key=f"{prefix}_max_articles",
    )

    tbl = _build_article_table(hits).head(int(max_articles))
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    download_table(tbl, filename_prefix=f"{prefix}_articles_{term.replace(' ', '_')}")


# -----------------------------
# Screen
# -----------------------------
def screen_live(
    freq: str,
    ngram_max: int,
    min_df: int,
    cloud_mode: str,
    window_days: int | None = None,     # lo que manda streamlit_app desde sidebar
    live_update_days: int | None = 14,  # compatibilidad
):
    df_raw, label, p = load_live_dataset()
    st.caption(f"{label} · {safe_last_update_label(p, 'Última actualización')}")

    if df_raw.empty:
        st.warning("No hay datos LIVE listos para mostrar.")
        return

    st.subheader("Explorar Live")

    # -----------------------------
    # NUEVO: Macro-área (filtro PLN)
    # -----------------------------
    area_options = ["Todas"] + sorted(MACRO_AREAS.keys())
    area_sel = st.selectbox("Macro-área (filtro PLN)", area_options, index=0, key="live_area_kw")

    # -----------------------------
    # NUEVO: Periodo (ventana LIVE) simple (presets)
    # -----------------------------
    preset_map = {
        "Usar slider del sidebar": None,
        "Últimos 7 días": 7
    }
    preset_label = st.selectbox(
        "Periodo (ventana Live)",
        list(preset_map.keys()),
        index=0,
        key="live_window_preset",
    )

    # Ventana LIVE (reciente)
    sidebar_days = window_days if (window_days is not None) else live_update_days
    days = preset_map[preset_label] if preset_map[preset_label] is not None else sidebar_days

    df0 = _apply_live_window(df_raw, days=days)
    df0 = _filter_by_macro_area(df0, area_sel)
    df0 = limit_df(df0, MAX_ROWS_TEXT)

    if df0.empty:
        st.warning("No hay registros LIVE con los filtros actuales (macro-área / periodo).")
        render_recent(df_raw.head(2000).copy() if not df_raw.empty else df_raw)
        return

    show_kpis(df0, freq)
    render_wordcloud(df0, "Nube de palabras (Live)", mode=cloud_mode)

    if period_count(df0, freq) < 2:
        st.info("Hay muy pocos periodos para ver tendencias. Prueba 'Semanas' o 'Meses'.")
        render_recent(df0)
        return

    # Candidatos -> matriz por periodo -> clasificación
    cand = pick_candidate_terms(df0, ngram_max=ngram_max, min_df=min_df)
    if cand.empty:
        st.warning("No se pudieron detectar temas. Prueba reducir 'Frecuencia mínima'.")
        return

    mat = build_term_matrix(df0, freq=freq, terms=tuple(cand["term"].tolist()))
    if mat.empty or mat.shape[0] < 2:
        st.warning("No se pudieron construir series por periodo.")
        return

    cls = classify_from_matrix(mat)
    if cls.empty:
        st.warning("No se pudieron clasificar tendencias.")
        return

    render_actions_header("Live")

    query = st.text_input("Buscar tema (opcional)", value="", key="live_search").strip().lower()

    def filtered(df_in: pd.DataFrame) -> pd.DataFrame:
        if not query:
            return df_in
        return df_in[df_in["term"].str.lower().str.contains(query, na=False)]

    # -----------------------------
    # Creciendo / Bajando
    # -----------------------------
    if st.session_state.action in ("creciendo", "bajando"):
        label_need = "creciendo" if st.session_state.action == "creciendo" else "bajando"
        st.subheader("Resultados")
        sub = filtered(cls[cls["label"] == label_need].copy())

        if sub.empty:
            st.info("No se encontraron temas con los filtros actuales.")
            return

        show = sub[["term", "total", "growth", "stability"]].rename(
            columns={"term": "tema", "total": "apariciones", "growth": "cambio", "stability": "estabilidad"}
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        download_table(show, filename_prefix=f"live_{label_need}")

        term = st.selectbox("Tema para ver su tendencia", sub["term"].head(250).tolist(), key="live_term_pick")
        series = mat[term].astype(float)
        periods = series.index.astype(str).tolist()

        fig = plot_trend_periods(periods, series.values, title=f"Live · Tendencia: {term}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.download_button(
            "Descargar serie del tema (CSV)",
            data=pd.DataFrame({"periodo": periods, "apariciones": series.values}).to_csv(index=False).encode("utf-8"),
            file_name=f"live_serie_{term.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        _render_articles_section(df0, term, prefix="live")

    # -----------------------------
    # Comparar
    # -----------------------------
    elif st.session_state.action == "comparar":
        st.subheader("Comparar dos temas")
        pool = filtered(cls.copy()).sort_values(["total"], ascending=False)
        terms = pool["term"].head(400).tolist()
        if len(terms) < 2:
            st.info("No hay suficientes temas para comparar.")
            return

        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Tema A", terms, index=0, key="live_cmp_a")
        with c2:
            b = st.selectbox("Tema B", terms, index=1, key="live_cmp_b")

        sa = mat[a].astype(float)
        sb = mat[b].astype(float)
        periods = sa.index.astype(str).tolist()

        fig, ax = plt.subplots(figsize=(10.8, 4.2))
        x = np.arange(len(periods))
        ax.plot(x, sa.values, label=a)
        ax.plot(x, sb.values, label=b)
        ax.set_title("Live · Comparación de tendencias")
        ax.set_xlabel("Periodo")
        ax.set_ylabel("Apariciones")
        ax.grid(True, alpha=0.25)

        n = len(periods)
        step = tick_step(n)
        tick_pos = x[::step]
        tick_lab = [periods[i] for i in range(0, n, step)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=30, ha="right")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Ver artículos relacionados (opcional)", expanded=False):
            pick = st.selectbox("Elegir tema para ver artículos", [a, b], key="live_cmp_articles_pick")
            _render_articles_section(df0, pick, prefix="live_cmp")

    # -----------------------------
    # Predicción
    # -----------------------------
    else:
        st.subheader("Predicción")
        pool = filtered(cls.copy()).sort_values(["label", "total"], ascending=[True, False])
        term_list = pool["term"].head(300).tolist()
        if not term_list:
            st.info("No hay temas disponibles para predecir con los filtros actuales.")
            return

        term = st.selectbox("Tema a predecir", term_list, key="live_pred_term")
        series = mat[term].astype(float)
        periods = series.index.astype(str).tolist()
        y = series.values

        c1, c2 = st.columns(2)
        with c1:
            h = st.slider("Periodos hacia adelante", 3, 12, 6, key="live_h")
        with c2:
            sp = st.slider("Patrón repetitivo (opcional)", 0, 24, 0, key="live_sp")

        fig, model_name = plot_forecast(periods, y, h=int(h), sp=int(sp), title=f"Live · Predicción: {term}")
        st.caption(f"Modelo: {model_name}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        _render_articles_section(df0, term, prefix="live_pred")
