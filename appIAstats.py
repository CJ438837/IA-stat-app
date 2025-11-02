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
    data_dict = pd.read_excel(uploaded_file, sheet_name=None)
    st.success(f"✅ Fichier importé : {uploaded_file.name}")
    
    # Sélection de la première feuille
    sheet_name = list(data_dict.keys())[0]
    df_sheet = data_dict[sheet_name]

    # --- Description de l’étude ---
    description = st.text_area("Décris ton étude en quelques phrases :", "")
    
    if st.button("Analyser"):
        if not description.strip():
            st.warning("Merci de décrire brièvement ton étude avant de lancer l'analyse.")
        else:
            # --- Extraction et traduction simplifiée des mots-clés ---
            tokens = re.findall(r'\b\w+\b', description.lower())
            stopwords_fr = set([
                "le","la","les","un","une","des","de","du","et","en","au","aux","avec",
                "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où",
                "se","sa","son","que","qui","ne","pas","plus","moins","comme","donc"
            ])
            keywords_fr = [w for w in tokens if w not in stopwords_fr]

            # Simple dictionnaire FR->EN pour la traduction
            translation_dict = {
                "poids": "weight",
                "taille": "height",
                "âge": "age",
                "taille": "size",
                "genre": "gender",
                "temps": "time"
            }
            keywords_en = [translation_dict.get(w, w) for w in keywords_fr]

            st.markdown("### 🧩 Mots-clés extraits")
            st.write(f"**Français :** {keywords_fr}")
            st.write(f"**Anglais :** {keywords_en}")

            # --- Import des fonctions IA-Stat ---
            from IA_STAT_descriptive_251125 import descriptive_analysis
            from IA_STAT_distribution_251125 import advanced_distribution_analysis
            from IA_STAT_interactif2 import propose_tests_interactif

            # --- Fonction adaptée pour DataFrame déjà chargé ---
            import numpy as np
            def detect_variable_types_df(df):
                results = []
                for col in df.columns:
                    col_data = df[col].dropna()
                    if col_data.empty:
                        continue

                    unique_vals = pd.Series(col_data).astype(str).str.strip().unique()
                    n_unique = len(unique_vals)

                    if n_unique == 2:
                        var_type = "binaire"
                    elif np.issubdtype(col_data.dtype, np.number):
                        var_type = "numérique"
                    else:
                        var_type = "catégorielle"

                    results.append({
                        "variable": col,
                        "type": var_type,
                        "valeurs_uniques": n_unique,
                        "exemples": unique_vals[:5]
                    })

                types_df = pd.DataFrame(results)
                return {"data": types_df}, {"data": df}

            # --- Détection des types ---
            types_dict, data_dict_adapted = detect_variable_types_df(df_sheet)
            types_df = types_dict['data']
            df_sheet = data_dict_adapted['data']

            # --- Analyse descriptive ---
            summary = descriptive_analysis(df_sheet, types_df)
            st.markdown("### 📊 Analyse descriptive")
            for var, stats in summary.items():
                st.write(f"**{var}** :")
                st.json(stats)

            # --- Analyse de distribution avancée ---
            st.markdown("### 📈 Analyse de distribution")
            distribution_df = advanced_distribution_analysis(df_sheet, types_df, output_folder="./plots")
            st.dataframe(distribution_df)

            # --- Tests statistiques interactifs ---
            st.markdown("### 🧮 Tests statistiques suggérés")
            propose_tests_interactif(types_df, distribution_df, df_sheet, keywords_en)

else:
    st.info("💡 Importez un fichier Excel pour commencer.")
