# src/state.py
from __future__ import annotations
import streamlit as st

def init_state():
    if "screen" not in st.session_state:
        st.session_state.screen = "menu"       # menu | historico | live
    if "action" not in st.session_state:
        st.session_state.action = "creciendo"  # creciendo | bajando | prediccion | comparar

    # Histórico: control de filtros
    if "hist_preset" not in st.session_state:
        st.session_state.hist_preset = "Todo"
    if "hist_start" not in st.session_state:
        st.session_state.hist_start = None
    if "hist_end" not in st.session_state:
        st.session_state.hist_end = None
    if "hist_filter_sig" not in st.session_state:
        st.session_state.hist_filter_sig = ""
