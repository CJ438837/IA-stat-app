import streamlit as st

st.title("📈 Résultats des tests")

if "last_result" in st.session_state:
    st.write(st.session_state["last_result"])
else:
    st.info("Aucun test exécuté pour le moment.")
