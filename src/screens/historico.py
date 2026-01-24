# src/screens/historico.py
from __future__ import annotations

import re
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import MaxNLocator

from src.analytics.trends_engine import build_term_matrix, classify_from_matrix, pick_candidate_terms
from src.config import AREA_TOP, AREA_TS, HIST_YEARS_KEEP, MAX_ROWS_TEXT
from src.data.datasets import load_historico_dataset
from src.data.io import file_exists, safe_last_update_label
from src.metrics import limit_df, period_count
from src.plotting.charts import plot_forecast, plot_trend_periods, tick_step
from src.taxonomy import MACRO_AREAS
from src.ui.widgets import download_table, render_actions_header, render_recent, show_kpis
from src.ui.wordcloud_ui import render_wordcloud


# -------------------------
# Helpers: macro-areas cache
# -------------------------
@st.cache_data(show_spinner=False, ttl=300)
def _load_area_aggs_cached(mtime_a: float, mtime_b: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.read_parquet(str(AREA_TS)) if file_exists(AREA_TS) else pd.DataFrame()
    top = pd.read_parquet(str(AREA_TOP)) if file_exists(AREA_TOP) else pd.DataFrame()
    return ts, top


def _get_area_aggs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not file_exists(AREA_TS) or not file_exists(AREA_TOP):
        return pd.DataFrame(), pd.DataFrame()
    return _load_area_aggs_cached(AREA_TS.stat().st_mtime, AREA_TOP.stat().st_mtime)


def render_macro_areas_historico():
    """
    Renderiza agregados precalculados por macro-área (si existen).
    Esto NO recalcula el histórico; solo consume artefactos AREA_TS y AREA_TOP.
    """
    ts, top = _get_area_aggs()
    if ts.empty:
        return

    with st.expander("Macro-áreas de Computación (Histórico)", expanded=False):
        st.caption("Basado en artefactos agregados (rápido, sin recalcular pesado).")

        ts2 = ts.copy()
        if "period" in ts2.columns:
            ts2["period"] = pd.to_datetime(ts2["period"], errors="coerce")

        if "area" not in ts2.columns:
            st.info("Agregado sin columna 'area'.")
            return

        areas = sorted(ts2["area"].dropna().unique().tolist())
        if not areas:
            st.info("No hay áreas disponibles.")
            return

        area = st.selectbox("Área (agregados)", areas, index=0, key="hist_area_agg_select")

        sub = ts2[ts2["area"] == area].sort_values("period")
        if sub.empty:
            st.info("Área sin datos.")
            return

        y_col = "rel_docs" if "rel_docs" in sub.columns else ("count_docs" if "count_docs" in sub.columns else None)
        if y_col is None:
            st.info("Agregado no trae 'rel_docs' ni 'count_docs'.")
            return

        periods = [p.strftime("%Y-%m-%d") for p in pd.to_datetime(sub["period"], errors="coerce")]
        y = sub[y_col].astype(float).values

        fig, ax = plt.subplots(figsize=(10.8, 4.2))
        ax.plot(np.arange(len(y)), y)
        ax.set_title(f"Evolución del área (Histórico): {area}")
        ax.set_xlabel("Periodo")
        ax.set_ylabel("Proporción de docs" if y_col == "rel_docs" else "Docs")
        ax.grid(True, alpha=0.25)

        n = len(periods)
        step = tick_step(n)
        tick_pos = np.arange(n)[::step]
        tick_lab = [periods[i] for i in range(0, n, step)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=30, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if not top.empty and "area" in top.columns:
            sub_top = top[top["area"] == area].copy()
            if not sub_top.empty:
                st.subheader("Términos destacados del área (agregados)")
                show_cols = [c for c in ["term", "score", "count", "growth"] if c in sub_top.columns]
                if show_cols:
                    st.dataframe(sub_top[show_cols].head(50), use_container_width=True, hide_index=True)


# -------------------------
# Filtering (time + PLN area)
# -------------------------
def _apply_year_cap(df: pd.DataFrame, years_keep: int) -> pd.DataFrame:
    """
    Recorta histórico a los últimos N años (por fecha máxima presente).
    """
    if df.empty or years_keep <= 0:
        return df
    mx = df["date"].max()
    if pd.isna(mx):
        return df
    cut = mx - pd.Timedelta(days=int(years_keep) * 365)
    return df[df["date"] >= cut].copy()


def _apply_date_filter(df: pd.DataFrame, preset: str, start_d, end_d) -> pd.DataFrame:
    if df.empty:
        return df
    mx = df["date"].max()
    if pd.isna(mx):
        return df

    if preset == "Últimos 30 días":
        cut = mx - pd.Timedelta(days=30)
        return df[df["date"] >= cut].copy()

    if preset == "Últimos 6 meses":
        cut = mx - pd.Timedelta(days=183)
        return df[df["date"] >= cut].copy()

    if preset == "Último año":
        cut = mx - pd.Timedelta(days=365)
        return df[df["date"] >= cut].copy()

    if preset == "Rango personalizado":
        out = df
        if start_d is not None:
            out = out[out["date"] >= pd.to_datetime(start_d)]
        if end_d is not None:
            out = out[out["date"] <= (pd.to_datetime(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
        return out.copy()

    return df.copy()


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
    # OR de keywords escapadas
    return "|".join([re.escape(k) for k in kws])


def _filter_by_macro_area(df: pd.DataFrame, area: str) -> pd.DataFrame:
    """
    Filtra por macro-área usando keywords (PLN básico) sobre df['text'].
    """
    if df.empty or not area or area == "Todas":
        return df

    pat = _compile_area_pattern(area)
    if not pat:
        return df

    # regex=True para OR; case-insensitive; NaN-safe
    mask = df["text"].astype(str).str.contains(pat, case=False, na=False, regex=True)
    return df[mask].copy()


# -------------------------
# Main screen
# -------------------------
def screen_historico(freq: str, ngram_max: int, min_df: int, cloud_mode: str):
    """
    Pantalla Histórico:
    - Recorta a últimos HIST_YEARS_KEEP años (default: 10)
    - Permite filtro PLN por macro-área (keywords)
    - Limita carga para UI (MAX_ROWS_TEXT)
    - Calcula candidatos TF-IDF + matrix por periodo + clasificación
    """
    df_raw, label, p = load_historico_dataset()
    st.caption(f"{label} · {safe_last_update_label(p, 'Última actualización')}")

    if df_raw.empty:
        st.warning("No hay datos históricos listos para mostrar.")
        return

    # 1) Cap duro de años (robusto contra dataset gigantes)
    df_raw = _apply_year_cap(df_raw, int(HIST_YEARS_KEEP))

    st.subheader("Explorar histórico")

    # 2) Selector macro-área PLN (keywords)
    area_options = ["Todas"] + sorted(MACRO_AREAS.keys())
    area_sel = st.selectbox("Macro-área (filtro PLN)", area_options, index=0, key="hist_area_kw")

    # 3) Presets de tiempo
    presets = ["Todo", "Últimos 30 días", "Últimos 6 meses", "Último año", "Rango personalizado"]

    if "hist_preset" not in st.session_state:
        st.session_state.hist_preset = "Todo"
    if "hist_start" not in st.session_state:
        st.session_state.hist_start = None
    if "hist_end" not in st.session_state:
        st.session_state.hist_end = None

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        preset = st.selectbox(
            "Periodo",
            presets,
            index=presets.index(st.session_state.hist_preset) if st.session_state.hist_preset in presets else 0,
            key="hist_preset_select",
        )

    if preset != st.session_state.hist_preset:
        st.session_state.hist_preset = preset
        if preset != "Rango personalizado":
            st.session_state.hist_start = None
            st.session_state.hist_end = None

    with c2:
        if preset == "Rango personalizado":
            st.session_state.hist_start = st.date_input("Desde", value=st.session_state.hist_start, key="hist_start_input")

    with c3:
        if preset == "Rango personalizado":
            st.session_state.hist_end = st.date_input("Hasta", value=st.session_state.hist_end, key="hist_end_input")

    # Firma de filtros (útil para depurar estado)
    st.session_state.hist_filter_sig = (
        f"{preset}|{st.session_state.hist_start}|{st.session_state.hist_end}|{area_sel}|{freq}|{ngram_max}|{min_df}"
    )

    # 4) Aplicar filtros
    df0 = _apply_date_filter(df_raw, preset, st.session_state.hist_start, st.session_state.hist_end)
    df0 = _filter_by_macro_area(df0, area_sel)

    # 5) Limitar para UI (evitar que normalize/cálculos revienten RAM)
    df0 = limit_df(df0, MAX_ROWS_TEXT)

    if df0.empty:
        st.warning("No hay registros con los filtros actuales (periodo/macro-área).")
        render_recent(df_raw.head(2000).copy() if not df_raw.empty else df_raw)
        return

    # KPIs + extras
    show_kpis(df0, freq)
    render_macro_areas_historico()
    render_wordcloud(df0, "Nube de palabras", mode=cloud_mode)

    # Validación de periodos
    if period_count(df0, freq) < 2:
        st.info("Hay muy pocos periodos para ver tendencias. Prueba 'Semanas' o 'Meses'.")
        render_recent(df0)
        return

    # 6) TF-IDF candidatos
    cand = pick_candidate_terms(df0, ngram_max=ngram_max, min_df=min_df)
    if cand.empty:
        st.warning("No se pudieron detectar temas. Prueba reducir 'Frecuencia mínima' o bajar n-gramas.")
        return

    # 7) Matriz por periodo
    mat = build_term_matrix(df0, freq=freq, terms=tuple(cand["term"].tolist()))
    if mat.empty or mat.shape[0] < 2:
        st.warning("No se pudieron construir series por periodo.")
        return

    # 8) Clasificación
    cls = classify_from_matrix(mat)
    if cls.empty:
        st.warning("No se pudieron clasificar tendencias.")
        return

    # Acción UI
    render_actions_header("Histórico")

    query = st.text_input("Buscar tema (opcional)", value="", key="hist_search").strip().lower()

    def _filtered(df_in: pd.DataFrame) -> pd.DataFrame:
        if not query:
            return df_in
        return df_in[df_in["term"].astype(str).str.lower().str.contains(query, na=False)].copy()

    # -------------------------
    # Actions
    # -------------------------
    action = getattr(st.session_state, "action", "prediccion")

    if action in ("creciendo", "bajando"):
        label_need = "creciendo" if action == "creciendo" else "bajando"
        st.subheader("Resultados")
        sub = _filtered(cls[cls["label"] == label_need].copy())

        if sub.empty:
            st.info("No se encontraron temas con los filtros actuales.")
            return

        show = sub[["term", "total", "growth", "stability"]].rename(
            columns={"term": "tema", "total": "apariciones", "growth": "cambio", "stability": "estabilidad"}
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        download_table(show, filename_prefix=f"historico_{label_need}")

        term = st.selectbox("Tema para ver su tendencia", sub["term"].head(250).tolist(), key="hist_term_pick")
        series = mat[term].astype(float)
        periods = series.index.astype(str).tolist()

        fig = plot_trend_periods(periods, series.values, title=f"Tendencia: {term}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.download_button(
            "Descargar serie del tema (CSV)",
            data=pd.DataFrame({"periodo": periods, "apariciones": series.values}).to_csv(index=False).encode("utf-8"),
            file_name=f"historico_serie_{term.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    elif action == "comparar":
        st.subheader("Comparar dos temas")
        pool = _filtered(cls.copy()).sort_values(["total"], ascending=False)
        terms = pool["term"].head(400).tolist()
        if len(terms) < 2:
            st.info("No hay suficientes temas para comparar.")
            return

        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Tema A", terms, index=0, key="hist_cmp_a")
        with c2:
            b = st.selectbox("Tema B", terms, index=1, key="hist_cmp_b")

        sa = mat[a].astype(float)
        sb = mat[b].astype(float)
        periods = sa.index.astype(str).tolist()

        fig, ax = plt.subplots(figsize=(10.8, 4.2))
        x = np.arange(len(periods))
        ax.plot(x, sa.values, label=a)
        ax.plot(x, sb.values, label=b)
        ax.set_title("Comparación de tendencias")
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

    else:
        st.subheader("Predicción")
        pool = _filtered(cls.copy()).sort_values(["label", "total"], ascending=[True, False])
        term_list = pool["term"].head(300).tolist()
        if not term_list:
            st.info("No hay temas disponibles para predecir con los filtros actuales.")
            return

        term = st.selectbox("Tema a predecir", term_list, key="hist_pred_term")
        series = mat[term].astype(float)
        periods = series.index.astype(str).tolist()
        y = series.values

        c1, c2 = st.columns(2)
        with c1:
            h = st.slider("Periodos hacia adelante", 3, 12, 6, key="hist_h")
        with c2:
            sp = st.slider("Patrón repetitivo (opcional)", 0, 24, 0, key="hist_sp")

        fig, model_name = plot_forecast(periods, y, h=int(h), sp=int(sp), title=f"Predicción: {term}")
        st.caption(f"Modelo: {model_name}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Cómo se calcula la predicción", expanded=False):
            st.write(
                "- Convertimos el texto en conteos por periodo.\n"
                "- Si hay suficiente señal, usamos un modelo estadístico (si está disponible).\n"
                "- Si no, usamos una tendencia simple con ajuste suave.\n"
                "- Sirve para ver dirección (sube/baja), no es una garantía."
            )
