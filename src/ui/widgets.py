# src/ui/widgets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st


# =========================
# Helpers (internos)
# =========================
def _safe_dt_minmax(df: pd.DataFrame) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df is None or df.empty or "date" not in df.columns:
        return None, None
    s = pd.to_datetime(df["date"], errors="coerce")
    if s.isna().all():
        return None, None
    return s.min(), s.max()


def _fmt_range(min_dt: Optional[pd.Timestamp], max_dt: Optional[pd.Timestamp]) -> str:
    if min_dt is None or max_dt is None:
        return "N/A"
    return f"{min_dt.date()} → {max_dt.date()}"


def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"


def _period_count_fallback(df: pd.DataFrame, freq: str) -> int:
    """Fallback si no quieres depender de src.metrics en widgets."""
    if df is None or df.empty or "date" not in df.columns:
        return 0
    s = pd.to_datetime(df["date"], errors="coerce").dropna()
    if s.empty:
        return 0
    # freq esperado: 'W' o 'M' (si te pasan 'Semanas'/'Meses' conviértelo antes en app)
    try:
        return s.dt.to_period(freq).nunique()
    except Exception:
        return 0


# =========================
# Widgets públicos (exports)
# =========================
def top_guide() -> None:
    """Caja superior de guía rápida (la usa streamlit_app.py)."""
    st.markdown("## Panel de Tendencias")
    with st.container():
        st.markdown(
            """
            <div style="
                background: rgba(220,220,220,0.9);
                color: #111;
                padding: 14px 16px;
                border-radius: 12px;
                margin-top: 4px;
                margin-bottom: 10px;
            ">
              <b>Guía rápida</b><br>
              1) Elige <b>Histórico</b> (largo plazo) o <b>Live</b> (reciente).<br>
              2) Usa <b>Creciendo</b>, <b>Bajando</b>, <b>Predicción</b> o <b>Comparar</b>.<br>
              3) Si algo sale vacío, normalmente el periodo elegido es muy corto o hay poca historia.
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_kpis(df: pd.DataFrame, freq: str) -> None:
    """KPIs: documentos, cobertura (min→max) y número de periodos."""
    min_dt, max_dt = _safe_dt_minmax(df)
    c1, c2, c3 = st.columns([1, 1, 0.6])

    with c1:
        st.markdown("**Documentos analizados**")
        st.markdown(f"<div style='font-size:36px; font-weight:700;'>{_fmt_int(len(df))}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("**Cobertura**")
        st.markdown(
            f"<div style='font-size:28px; font-weight:650;'>{_fmt_range(min_dt, max_dt)}</div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown("**Periodos**")
        pc = _period_count_fallback(df, freq)
        st.markdown(f"<div style='font-size:36px; font-weight:700;'>{_fmt_int(pc)}</div>", unsafe_allow_html=True)


def render_actions_header(scope_name: str) -> None:
    """
    Barra de acciones (Creciendo/Bajando/Predicción/Comparar)
    Guarda selección en st.session_state.action.
    """
    if "action" not in st.session_state:
        st.session_state.action = "creciendo"

    st.markdown("---")
    st.markdown(f"### Acciones ({scope_name})")

    labels = [("creciendo", "Creciendo"), ("bajando", "Bajando"), ("prediccion", "Predicción"), ("comparar", "Comparar")]

    cols = st.columns(len(labels))
    for (key, label), col in zip(labels, cols):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.action = key

    st.caption(f"Acción actual: **{st.session_state.action}**")


def render_recent(df: pd.DataFrame, n: int = 20) -> None:
    """Tabla rápida de últimos registros."""
    if df is None or df.empty:
        st.info("Sin registros para mostrar.")
        return

    st.markdown("### Más reciente")
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.sort_values("date", ascending=False)

    show_cols = [c for c in ["date", "title", "text", "category", "source"] if c in tmp.columns]
    if not show_cols:
        show_cols = tmp.columns.tolist()[:5]

    st.dataframe(tmp[show_cols].head(n), use_container_width=True, hide_index=True)


def download_table(df: pd.DataFrame, filename_prefix: str = "tabla") -> None:
    """Botón para descargar un dataframe como CSV (UTF-8)."""
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar tabla (CSV)",
        data=csv_bytes,
        file_name=f"{filename_prefix}.csv",
        mime="text/csv",
    )


def system_status_box(**items) -> None:
    """
    Cajita de estado para sidebar u otra sección.
    Ejemplo:
      system_status_box(Pantalla="live", Runner=True, Cache_TTL=300)
    """
    with st.expander("Estado del sistema", expanded=False):
        if not items:
            st.write("Sin datos de estado.")
            return
        for k, v in items.items():
            st.write(f"- **{k}**: {v}")
