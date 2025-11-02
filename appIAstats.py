import streamlit as st
import pandas as pd
from IA_STAT_typevariable_251125 import detect_variable_types
from IA_STAT_descriptive_251125 import descriptive_analysis
from IA_STAT_distribution_251125 import advanced_distribution_analysis
from IA_STAT_interactif2 import propose_tests_interactif  # la fonction complète que nous avons réécrite

st.set_page_config(page_title="🧠 Analyse statistique automatisée", layout="wide")

st.title("🧠 Analyse statistique automatisée")
st.write("Importe ton fichier Excel pour générer une analyse descriptive et des visualisations automatiquement.")

# --- Téléversement de fichier ---
uploaded_file = st.file_uploader("📂 Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Lecture du fichier Excel complet
    data_dict = pd.read_excel(uploaded_file, sheet_name=None)
    st.success(f"✅ Fichier importé : {uploaded_file.name}")

    # Sélection de la première feuille
    sheet_name = list(data_dict.keys())[0]
    df_sheet = data_dict[sheet_name]

    # --- Description de l’étude ---
    description = st.text_area("Décris ton étude en quelques phrases :", "")

    if st.button("Analyser"):
        if df_sheet.empty:
            st.warning("Le fichier Excel est vide.")
        else:
            # --- Détection des types de variables ---
            try:
                # On passe directement le DataFrame à la fonction adaptée
                types_dict, data_dict_adapted = detect_variable_types(df_sheet)
                types_df = types_dict[list(types_dict.keys())[0]]
                df_sheet = data_dict_adapted[list(data_dict_adapted.keys())[0]]
            except Exception as e:
                st.error(f"Erreur lors de la détection des types : {e}")
                st.stop()

            # --- Analyse descriptive ---
            try:
                summary = descriptive_analysis(df_sheet, types_df)
                st.markdown("### 📊 Analyse descriptive")
                for var, stats_dict in summary.items():
                    st.write(f"**{var}** :")
                    st.json(stats_dict)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse descriptive : {e}")

            # --- Analyse de distribution avancée ---
            try:
                st.markdown("### 📈 Analyse de distribution")
                distribution_df = advanced_distribution_analysis(df_sheet, types_df, output_folder="./plots")
                st.dataframe(distribution_df)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de distribution : {e}")

            # --- Tests statistiques interactifs ---
            try:
                st.markdown("### 🧮 Tests statistiques interactifs")
                # On ne fait plus de traduction, on passe directement la liste de mots-clés
                mots_cles = description.lower().split()
                propose_tests_interactif(types_df, distribution_df, df_sheet, mots_cles)
            except Exception as e:
                st.error(f"Erreur lors des tests interactifs : {e}")

else:
    st.info("💡 Importez un fichier Excel pour commencer.")
