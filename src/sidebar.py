# src/sidebar.py
from __future__ import annotations

import sys
import time
from typing import Any, Dict, Tuple, List

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


def _clear_streamlit_cache() -> None:
    """Limpia cache de Streamlit (si existe en la versión instalada)."""
    try:
        st.cache_data.clear()
    except Exception:
        pass


def _run_live_update(
    days_back: int,
    api_page_size: int,
    api_max_total: int,
    timeout_sec: int = 1800,
) -> Tuple[int, str, List[str]]:
    """
    Ejecuta el runner LIVE como módulo (evita problemas de imports con 'src').
    Devuelve: (returncode, salida_combinada, cmd)
    """
    days_back = int(days_back)
    api_page_size = int(api_page_size)
    api_max_total = int(api_max_total)

    cmd = [
        sys.executable,
        "-m",
        "src.live_runner",
        "--days-back",
        str(days_back),
        "--log-level",
        "INFO",
        # nuevos knobs (deben existir en tu live_runner CLI)
        "--api-page-size",
        str(api_page_size),
        "--api-max-total",
        str(api_max_total),
    ]

    code, out = run_script_capture(cmd, timeout_sec=timeout_sec)
    return int(code), (out or ""), cmd


def render_sidebar() -> Dict[str, Any]:
    # -----------------------------
    # Navegación
    # -----------------------------
    st.header("Navegación")
    if st.button("Menú principal", key="sb_menu"):
        st.session_state.screen = "menu"
        st.rerun()

    st.divider()

    # -----------------------------
    # Modo Expo (estabilidad)
    # -----------------------------
    st.subheader("Modo")
    expo_mode = st.toggle(
        "Modo Expo (estable)",
        value=True,
        help="Fija parámetros recomendados para demo: evita vacíos y reduce riesgo de fallas.",
        key="sb_expo_mode",
    )

    st.divider()

    # -----------------------------
    # Opciones globales
    # -----------------------------
    st.subheader("Opciones")

    # Defaults recomendados según modo
    default_freq_index = 0 if expo_mode else 2  # Semanas en expo, Días fuera
    freq_label = st.selectbox(
        "Ver cambios por",
        ["Semanas", "Meses", "Días"],
        index=default_freq_index,
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
        # En expo conviene ngram=2 y min_df=2 para resultados estables
        ngram_default = 1  # index=1 -> valor 2
        min_df_default = 2

        ngram_max = st.selectbox(
            "Detectar frases de",
            [1, 2, 3],
            index=(ngram_default if expo_mode else 1),
            key="sb_ngram",
        )
        min_df = st.slider(
            "Frecuencia mínima",
            1,
            10,
            (min_df_default if expo_mode else 2),
            key="sb_min_df",
        )

        # Ventana de visualización para LIVE (solo cuando estás en Live)
        if st.session_state.screen == "live":
            # En expo, 30-90 días suele ser perfecto para “Semanas”
            window_days = st.slider(
                "Días a revisar (Live)",
                7,
                365,
                (60 if expo_mode else 180),
                key="sb_live_days",
            )
        else:
            window_days = (60 if expo_mode else 180)

        # Ventana para el refresco de LIVE (runner)
        live_update_days = st.slider(
            "Actualizar Live con ventana (días)",
            7,
            365,
            (30 if expo_mode else 180),
            key="sb_live_update_days",
        )

        # Knobs de ingesta API (para volumen)
        api_page_size = st.selectbox(
            "API page size",
            [50, 100, 200],
            index=2,  # 200
            key="sb_api_page_size",
            help="Tamaño de página para arXiv API. 200 suele ser eficiente.",
        )
        api_max_total = st.selectbox(
            "API max total",
            [1000, 2000, 4000, 6000],
            index=(2 if expo_mode else 1),  # 4000 en expo, 2000 fuera
            key="sb_api_max_total",
            help="Máximo total de registros a traer en la actualización Live.",
        )

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
        st.write(f"Modo Expo: {'Sí' if expo_mode else 'No'}")

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

    # Guarda logs de la última ejecución en sesión (para ver aunque haya sido OK)
    if "live_last_run" not in st.session_state:
        st.session_state.live_last_run = {"code": None, "cmd": None, "out": ""}

    if st.button("Actualizar Live", key="sb_live_update"):
        with st.spinner("Actualizando Live…"):
            code, out, cmd = _run_live_update(
                days_back=live_update_days,
                api_page_size=api_page_size,
                api_max_total=api_max_total,
                timeout_sec=1800,
            )

        st.session_state.live_last_run = {"code": code, "cmd": cmd, "out": out}

        if code == 0:
            st.success("Live actualizado correctamente.")
            _clear_streamlit_cache()
            time.sleep(0.2)
            st.rerun()
        else:
            st.error("Falló la actualización de Live. Revisa los logs abajo.")

    # Logs visibles siempre (útil para expo)
    with st.expander("Logs de actualización Live", expanded=False):
        lr = st.session_state.get("live_last_run", {}) or {}
        code = lr.get("code")
        cmd = lr.get("cmd")
        out = lr.get("out", "")

        st.write(f"Return code: {code}")
        if cmd:
            st.caption("Comando:")
            st.code(" ".join(cmd))
        if out:
            st.caption("Salida / error:")
            st.text(out[:12000])
        else:
            st.caption("Sin logs aún.")

    if st.button("Recargar datos", key="sb_reload"):
        _clear_streamlit_cache()
        st.rerun()

    return dict(
        freq=freq,
        cloud_mode=cloud_mode,
        ngram_max=int(ngram_max),
        min_df=int(min_df),
        window_days=int(window_days),
        live_update_days=int(live_update_days),
    )
