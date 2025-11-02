import streamlit as st
import pandas as pd
import re
from Bio import Entrez

# --- Configuration PubMed ---
Entrez.email = "ton.email@example.com"

st.set_page_config(page_title="🧠 IA Statistique", layout="wide")
st.title("🧠 Analyse statistique automatisée")
st.write("Importe ton fichier Excel pour générer automatiquement une analyse descriptive, des visualisations et des tests interactifs.")

# --- Téléversement de fichier ---
uploaded_file = st.file_uploader("📂 Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Lecture du fichier
    try:
        data_dict = pd.read_excel(uploaded_file, sheet_name=None)
        st.success(f"✅ Fichier importé : {uploaded_file.name}")
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
        st.stop()

    # Sélection de la première feuille
    sheet_name = list(data_dict.keys())[0]
    df_sheet = data_dict[sheet_name]

    # --- Description de l’étude ---
    description = st.text_area("Décris ton étude en quelques phrases :", "")

    if st.button("Analyser"):
        if not description.strip():
            st.warning("Merci de décrire brièvement ton étude avant de lancer l'analyse.")
            st.stop()

        # --- Extraction des mots-clés ---
        tokens = re.findall(r'\b\w+\b', description.lower())
        stopwords_fr = set([
            "le","la","les","un","une","des","de","du","et","en","au","aux","avec",
            "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où",
            "se","sa","son","que","qui","ne","pas","plus","moins","comme","donc"
        ])
        keywords_fr = [w for w in tokens if w not in stopwords_fr]

        st.markdown("### 🧩 Mots-clés extraits")
        st.write(f"**Français :** {keywords_fr}")

        # --- Import des fonctions IA-Stat ---
        from IA_STAT_typevariable_251125 import detect_variable_types
        from IA_STAT_descriptive_251125 import descriptive_analysis
        from IA_STAT_distribution_251125 import advanced_distribution_analysis
        from IA_STAT_interactif2 import propose_tests_interactif
        from IA_STAT_Illustrations_251125 import plot_descriptive

        # --- Détection des types ---
        types_dict, data_dict_adapted = detect_variable_types(df_sheet)
        types_df = types_dict[sheet_name]
        df_sheet = data_dict_adapted[sheet_name]

        # --- Analyse descriptive ---
        st.markdown("### 📊 Analyse descriptive")
        summary = descriptive_analysis(df_sheet, types_df)
        for var, stats in summary.items():
            st.write(f"**{var}** :")
            st.json(stats)

        # --- Visualisations descriptives ---
        st.markdown("### 📈 Visualisations descriptives")
        try:
            plot_descriptive(df_sheet, types_df, output_folder="./plots_streamlit")
            st.success("✅ Graphiques descriptifs générés dans ./plots_streamlit")
        except Exception as e:
            st.warning(f"Impossible de générer tous les graphiques : {e}")

        # --- Analyse de distribution avancée ---
        st.markdown("### 📊 Analyse distribution avancée")
        try:
            distribution_df = advanced_distribution_analysis(df_sheet, types_df, output_folder="./plots_streamlit")
            st.dataframe(distribution_df)
        except Exception as e:
            st.warning(f"Impossible de générer l'analyse de distribution : {e}")
            distribution_df = pd.DataFrame()

        # --- Tests statistiques interactifs ---
        st.markdown("### 🧮 Tests statistiques interactifs")
        try:
            propose_tests_interactif(types_df, distribution_df, df_sheet, keywords_fr)
        except Exception as e:
            st.warning(f"Erreur lors des tests interactifs : {e}")

else:
    st.info("💡 Importez un fichier Excel pour commencer.")
