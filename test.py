import re
from googletrans import Translator
from Bio import Entrez
import pandas as pd

# --- Config PubMed ---
Entrez.email = "ton.email@example.com"

# --- Entrée utilisateur ---
description = input("Décrivez votre étude en quelques phrases : ")

# --- Extraction de mots alphabétiques ---
tokens = re.findall(r'\b\w+\b', description.lower())
stopwords_fr = set([
    "le","la","les","un","une","des","de","du","et","en","au","aux","avec",
    "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où","se",
    "sa","son","que","qui","ne","pas","plus","moins","comme","donc"
])
keywords_fr = [w for w in tokens if w not in stopwords_fr]

# --- Traduction en anglais ---
translator = Translator()
keywords_en = [translator.translate(word, src='fr', dest='en').text for word in keywords_fr]

print(f"\nMots-clés français : {keywords_fr}")
print(f"Mots-clés traduits anglais : {keywords_en}")

# --- Import des fonctions locales ---
from IA_STAT_typevariable_251125 import detect_variable_types
from IA_STAT_descriptive_251125 import descriptive_analysis
from IA_STAT_Illustrations_251125 import plot_descriptive
from IA_STAT_distribution_251125 import advanced_distribution_analysis
from IA_STAT_interactif2 import propose_tests_interactif

# --- Chemin du fichier Excel à analyser ---
file_path = "C:/Users/cedri/Downloads/Poids_Decembre_AV.xlsx"

# --- Détection des types de variables ---
types_dict, data_dict = detect_variable_types(file_path)
sheet_name = list(types_dict.keys())[0]  # première feuille par défaut
types_df = types_dict[sheet_name]
df_sheet = data_dict[sheet_name]

# --- Analyse descriptive ---
summary = descriptive_analysis(df_sheet, types_df)
print("\n=== Analyse descriptive ===")
for var, stats in summary.items():
    print(f"\n--- Variable : {var} ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

# --- Visualisations univariées et bivariées ---
output_folder_plots = "D:/Programation/IA stat/plots_test"
plot_descriptive(df_sheet, types_df, output_folder=output_folder_plots)
print(f"\nGraphiques générés dans le dossier : {output_folder_plots}")

# --- Analyse de distribution avancée ---
output_folder_distribution = "D:/Programation/IA stat/distribution_test"
distribution_df = advanced_distribution_analysis(df_sheet, types_df, output_folder=output_folder_distribution)
print("\n=== Résultat analyse distribution avancée ===")
print(distribution_df)
print(f"\nGraphiques de distribution générés dans le dossier : {output_folder_distribution}")

# --- Tests statistiques interactifs ---
propose_tests_interactif(types_df, distribution_df, df_sheet, keywords_en)
