import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }

        :root {
          --card-bg: rgba(255,255,255,0.72);
          --card-border: rgba(0,0,0,0.10);
          --muted: rgba(0,0,0,0.55);
        }
        [data-theme="dark"] {
          --card-bg: rgba(38,39,48,0.55);
          --card-border: rgba(255,255,255,0.10);
          --muted: rgba(255,255,255,0.55);
        }

        .card {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 14px;
            background: var(--card-bg);
        }

        .bigbtn button {
            width: 100%;
            height: 56px;
            font-size: 16px;
            border-radius: 14px;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--card-border);
            padding: 10px;
            border-radius: 12px;
            background: var(--card-bg);
        }

        .muted { color: var(--muted); font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
