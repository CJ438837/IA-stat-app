import streamlit as st
import pandas as pd
import re
import numpy as np
from IA_STAT_typevariable_251125 import detect_variable_types_df
from IA_STAT_descriptive_251125 import descriptive_analysis
from IA_STAT_distribution_251125 import advanced_distribution_analysis

st.title("🧠 Analyse statistique automatisée")

uploaded_file = st.file_uploader("📂 Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_dict = pd.read_excel(uploaded_file, sheet_name=None)
    sheet_name = list(df_dict.keys())[0]
    df = df_dict[sheet_name]

    description = st.text_area("Décris ton étude :", "")
    if st.button("Analyser"):
        if not description.strip():
            st.warning("Merci d’écrire une brève description.")
        else:
            tokens = re.findall(r'\b\w+\b', description.lower())
            stopwords_fr = set(["le","la","les","un","une","des","de","du","et","en","au","aux",
                                "pour","sur","dans","par","ce","ces","est","sont","ou","où",
                                "se","sa","son","que","qui","ne","pas","plus","moins","comme","donc"])
            keywords = [w for w in tokens if w not in stopwords_fr]

            types_dict, data_dict_adapted = detect_variable_types_df(df)
            types_df = types_dict['data']
            df = data_dict_adapted['data']

            summary = descriptive_analysis(df, types_df)
            distribution_df = advanced_distribution_analysis(df, types_df, output_folder="./plots")

            # Stocke les résultats pour les autres pages
            st.session_state["types_df"] = types_df
            st.session_state["distribution_df"] = distribution_df
            st.session_state["df"] = df
            st.session_state["keywords"] = keywords

            st.success("Analyse terminée ✅ — passe à l’onglet *Tests statistiques*")
else:
    st.info("💡 Importez un fichier Excel pour commencer.")
