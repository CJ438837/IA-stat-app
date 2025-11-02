import streamlit as st
import pandas as pd
import re
from Bio import Entrez

# === Configuration ===
Entrez.email = "ton.email@example.com"

st.title("🧠 Analyse statistique automatisée")
st.write("Importe ton fichier Excel pour générer une analyse descriptive et des visualisations automatiquement.")

# --- Téléversement de fichier ---
uploaded_file = st.file_uploader("📂 Importer un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Lecture du fichier
    df = pd.read_excel(uploaded_file, sheet_name=None)
    st.success(f"✅ Fichier importé : {uploaded_file.name}")
    sheet_name = list(df.keys())[0]
    df_sheet = df[sheet_name]
    
    # --- Description de l’étude ---
    description = st.text_area("Décris ton étude en quelques phrases :", "")
    if st.button("Analyser"):
        if not description.strip():
            st.warning("Merci de décrire brièvement ton étude avant de lancer l'analyse.")
        else:
            # --- Extraction et traduction des mots-clés ---
            tokens = re.findall(r'\b\w+\b', description.lower())
            stopwords_fr = set(["le","la","les","un","une","des","de","du","et","en","au","aux","avec",
                                "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où",
                                "se","sa","son","que","qui","ne","pas","plus","moins","comme","donc"])
            keywords_fr = [w for w in tokens if w not in stopwords_fr]

            st.markdown("### 🧩 Mots-clés extraits")
            st.write(f"**Français :** {keywords_fr}")
            st.write(f"**Anglais :** {keywords_en}")

            # --- Analyse automatisée ---
            from IA_STAT_typevariable_251125 import detect_variable_types
            from IA_STAT_descriptive_251125 import descriptive_analysis
            from IA_STAT_distribution_251125 import advanced_distribution_analysis
            from IA_STAT_interactif2 import propose_tests_interactif

            # Détection types
            types_dict, data_dict = detect_variable_types(uploaded_file)
            sheet_name = list(types_dict.keys())[0]
            types_df = types_dict[sheet_name]
            df_sheet = data_dict[sheet_name]

            # Analyse descriptive
            summary = descriptive_analysis(df_sheet, types_df)
            st.markdown("### 📊 Analyse descriptive")
            for var, stats in summary.items():
                st.write(f"**{var}** :")
                st.json(stats)

            # Distribution avancée
            st.markdown("### 📈 Analyse de distribution")
            distribution_df = advanced_distribution_analysis(df_sheet, types_df, output_folder="./plots")

            st.dataframe(distribution_df)

            # Tests statistiques
            st.markdown("### 🧮 Tests statistiques suggérés")
            propose_tests_interactif(types_df, distribution_df, df_sheet, keywords_en)

else:
    st.info("💡 Importez un fichier Excel pour commencer.")
