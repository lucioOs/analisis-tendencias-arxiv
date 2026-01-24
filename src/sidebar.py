# src/sidebar.py
from __future__ import annotations

import sys
import time
import streamlit as st

from src.config import (
    CACHE_TTL_SEC,
    MAX_ROWS_TEXT,
    LIVE_DATASET,
    AREA_TS,
    AREA_TOP,
)
from src.data.io import file_exists, safe_last_update_label, run_script_capture
from src.data.datasets import load_historico_dataset


def render_sidebar() -> dict:
    # -----------------------------
    # Navegación
    # -----------------------------
    st.header("Navegación")
    if st.button("Menú principal", key="sb_menu"):
        st.session_state.screen = "menu"
        st.rerun()

    st.divider()

    # -----------------------------
    # Opciones globales
    # -----------------------------
    st.subheader("Opciones")

    freq_label = st.selectbox(
        "Ver cambios por",
        ["Semanas", "Meses", "Días"],
        index=0,
        key="sb_freq_label",
    )
    freq = "W" if freq_label == "Semanas" else ("M" if freq_label == "Meses" else "D")

    cloud_mode = st.selectbox(
        "Tipo de nube",
        ["Destacados (TF-IDF)", "Frecuencia"],
        index=0,
        key="sb_cloud_mode",
    )

    # -----------------------------
    # Ajustes
    # -----------------------------
    with st.expander("Ajustes", expanded=False):
        ngram_max = st.selectbox("Detectar frases de", [1, 2, 3], index=1, key="sb_ngram")
        min_df = st.slider("Frecuencia mínima", 1, 10, 2, key="sb_min_df")

        # Ventana de visualización para LIVE (solo cuando estás en Live)
        if st.session_state.screen == "live":
            window_days = st.slider("Días a revisar (Live)", 7, 365, 180, key="sb_live_days")
        else:
            window_days = 180

        # Ventana para el refresco de LIVE (runner)
        live_update_days = st.slider("Actualizar Live con ventana (días)", 7, 365, 180, key="sb_live_update_days")

    st.divider()

    # -----------------------------
    # Estado del sistema
    # -----------------------------
    with st.expander("Estado del sistema", expanded=False):
        st.write(f"Pantalla: {st.session_state.screen}")
        st.write(f"Acción: {st.session_state.action}")
        st.write(
            "Agregados macro-áreas (Histórico): "
            f"{'Sí' if (file_exists(AREA_TS) and file_exists(AREA_TOP)) else 'No'}"
        )
        st.write(f"Cache TTL (s): {CACHE_TTL_SEC}")
        st.write(f"MAX_ROWS_TEXT: {MAX_ROWS_TEXT}")

        if st.session_state.screen == "historico":
            st.write(f"Filtro Histórico: {st.session_state.get('hist_filter_sig', '')}")

    st.divider()

    # -----------------------------
    # Información de datos
    # -----------------------------
    st.subheader("Datos")
    df_h, label_h, p_h = load_historico_dataset()
    st.caption(label_h)
    st.caption(safe_last_update_label(p_h, "Histórico"))
    st.caption(safe_last_update_label(LIVE_DATASET, "Live"))

    st.divider()

    # -----------------------------
    # Acciones LIVE
    # -----------------------------
    st.subheader("Live")

    if st.button("Actualizar Live", key="sb_live_update"):
        with st.spinner("Actualizando Live…"):
            # Ejecutar como módulo evita: ModuleNotFoundError: No module named 'src'
            cmd = [sys.executable, "-m", "src.live_runner", str(int(live_update_days))]
            code, out = run_script_capture(cmd, timeout_sec=1800)

        if code == 0:
            st.success("Live actualizado correctamente.")
            try:
                st.cache_data.clear()
            except Exception:
                pass
            time.sleep(0.2)
            st.rerun()
        else:
            st.error("Falló la actualización de Live.")
            st.text(out[:5000])

    if st.button("Recargar datos", key="sb_reload"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

    return dict(
        freq=freq,
        cloud_mode=cloud_mode,
        ngram_max=int(ngram_max),
        min_df=int(min_df),
        window_days=int(window_days),
        live_update_days=int(live_update_days),
    )
