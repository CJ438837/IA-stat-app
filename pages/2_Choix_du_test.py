import streamlit as st
from IA_STAT_interactif2 import propose_tests_interactif

st.title("🧮 Tests statistiques")

# Vérifie si les données existent déjà
if "df" in st.session_state:
    df = st.session_state["df"]
    types_df = st.session_state["types_df"]
    distribution_df = st.session_state["distribution_df"]
    keywords = st.session_state["keywords"]

    propose_tests_interactif(types_df, distribution_df, df, keywords)

    st.info("✅ Sélectionne les tests ci-dessus puis passe à l'onglet Résultats")
else:
    st.warning("⚠️ Merci de d’abord exécuter l’analyse descriptive dans la page principale.")
