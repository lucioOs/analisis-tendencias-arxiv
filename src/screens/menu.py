import streamlit as st


def screen_menu():
    st.subheader("Menú principal")
    st.write("¿Qué quieres explorar?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
        if st.button("Histórico", key="menu_hist"):
            st.session_state.screen = "historico"
            st.session_state.action = "creciendo"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="muted">Tendencias de largo plazo con el dataset del proyecto.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
        if st.button("Live", key="menu_live"):
            st.session_state.screen = "live"
            st.session_state.action = "creciendo"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="muted">Lo más reciente y actualizable.</div>', unsafe_allow_html=True)
