import streamlit as st
from IA_STAT_interactif2 import propose_tests_interactif

st.title("🔬 Choix du test statistique")

if "df" not in st.session_state:
    st.warning("⚠️ Veuillez d'abord importer et analyser un fichier sur la page principale.")
    st.stop()

df = st.session_state["df"]
types_df = st.session_state["types_df"]
distribution_df = st.session_state["distribution_df"]
keywords = st.session_state["keywords"]

st.info("🧩 Sélectionnez les tests que vous souhaitez exécuter.")

# Appel de la fonction de proposition de tests (interactive)
tests_selectionnes = propose_tests_interactif(types_df, distribution_df, df, keywords)

# Sauvegarde du choix
if st.button("➡️ Valider la sélection et passer à l'exécution"):
    st.session_state["tests_selectionnes"] = tests_selectionnes
    st.switch_page("pages/2_Résultats_du_test.py")
