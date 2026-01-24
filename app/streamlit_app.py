# app/streamlit_app.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# -------------------------------------------------------------------
# FIX CRÍTICO: permitir imports "src.*" aunque streamlit se ejecute desde /app
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # D:\Proyecto
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"

from src.ui.styles import apply_custom_css
from src.ui.widgets import top_guide
from src.screens.menu import screen_menu
from src.screens.historico import screen_historico
from src.screens.live import screen_live
from src.state import init_state
from src.config import APP_TITLE


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_custom_css()
    init_state()

    st.title(APP_TITLE)
    top_guide()

    # Sidebar global
    with st.sidebar:
        from src.sidebar import render_sidebar  # import diferido para evitar ciclos
        sidebar_cfg = render_sidebar()

    screen = st.session_state.screen

    if screen == "menu":
        screen_menu()
        return

    if screen == "historico":
        screen_historico(
            freq=sidebar_cfg["freq"],
            ngram_max=sidebar_cfg["ngram_max"],
            min_df=sidebar_cfg["min_df"],
            cloud_mode=sidebar_cfg["cloud_mode"],
        )
        return

    if screen == "live":
        screen_live(
            freq=sidebar_cfg["freq"],
            window_days=sidebar_cfg["window_days"],
            ngram_max=sidebar_cfg["ngram_max"],
            min_df=sidebar_cfg["min_df"],
            cloud_mode=sidebar_cfg["cloud_mode"],
            live_update_days=sidebar_cfg["live_update_days"],
        )
        return


if __name__ == "__main__":
    main()
