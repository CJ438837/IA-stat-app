import streamlit as st
import pandas as pd
import re
import numpy as np

from IA_STAT_typevariable_251125 import detect_variable_types_df
from IA_STAT_descriptive_251125 import descriptive_analysis
from IA_STAT_distribution_251125 import advanced_distribution_analysis

st.set_page_config(page_title="IA Stats", layout="wide")
st.title("🧠 Analyse statistique automatisée")

uploaded_file = st.file_uploader("📂 Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    data_dict = pd.read_excel(uploaded_file, sheet_name=None)
    sheet_name = list(data_dict.keys())[0]
    df = data_dict[sheet_name]
    st.success(f"✅ Fichier importé : {uploaded_file.name} (Feuille : {sheet_name})")

    description = st.text_area("🧾 Décris ton étude brièvement :", "")

    if st.button("Analyser le fichier"):
        if not description.strip():
            st.warning("⚠️ Merci de décrire ton étude avant de lancer l'analyse.")
            st.stop()

        # --- Extraction de mots-clés simples ---
        tokens = re.findall(r'\b\w+\b', description.lower())
        stopwords = {"le", "la", "les", "un", "une", "des", "de", "et", "en", "au", "aux", "avec", "pour", "dans", "par", "est"}
        keywords = [t for t in tokens if t not in stopwords]
        st.write(f"**Mots-clés détectés :** {keywords}")

        # --- Détection des types ---
        types_dict, data_dict_adapted = detect_variable_types_df(df)
        types_df = types_dict["data"]
        df = data_dict_adapted["data"]

        # --- Analyse descriptive ---
        st.markdown("### 📊 Analyse descriptive")
        summary = descriptive_analysis(df, types_df)
        for var, stats in summary.items():
            st.write(f"**{var}** :")
            st.json(stats)

        # --- Analyse de distribution ---
        st.markdown("### 📈 Analyse de distribution")
        distribution_df = advanced_distribution_analysis(df, types_df, output_folder="./plots")
        st.dataframe(distribution_df)

        # --- Stocker les objets en session pour la page suivante ---
        st.session_state["df"] = df
        st.session_state["types_df"] = types_df
        st.session_state["distribution_df"] = distribution_df
        st.session_state["keywords"] = keywords

        st.success("✅ Analyse terminée. Passez à la page « Choix du test ».")
else:
    st.info("💡 Importez un fichier Excel pour commencer.")


import streamlit as st

# --- Gestion de la navigation ---
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# --- Barre latérale simple ---
st.sidebar.title("Navigation")
if st.sidebar.button("Accueil"):
    go_to("Accueil")
if st.sidebar.button("Analyse descriptive"):
    go_to("Analyse descriptive")
if st.sidebar.button("Tests statistiques"):
    go_to("Tests statistiques")
if st.sidebar.button("Résultats"):
    go_to("Résultats")

# --- Contenu selon la page ---
if st.session_state.page == "Accueil":
    st.title("🧠 Accueil")
    st.write("Bienvenue dans ton assistant IA-Stat.")
    if st.button("Commencer l'analyse"):
        go_to("Analyse descriptive")

elif st.session_state.page == "Analyse descriptive":
    st.title("📊 Analyse descriptive")
    st.write("Page pour les analyses descriptives.")
    if st.button("Passer aux tests statistiques"):
        go_to("Tests statistiques")

elif st.session_state.page == "Tests statistiques":
    st.title("🧮 Tests statistiques")
    st.write("Sélectionne les tests à exécuter.")
    if st.button("Voir les résultats"):
        go_to("Résultats")

elif st.session_state.page == "Résultats":
    st.title("📈 Résultats")
    st.write("Les résultats de tes tests apparaîtront ici.")
