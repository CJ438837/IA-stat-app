import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def run_test_interactif(tests_df, df):
    """
    Parcourt chaque test proposé et demande à l'utilisateur s'il veut l'exécuter.
    """
    for idx, row in tests_df.iterrows():
        print("\n--- Suggestion ---")
        print(f"Test proposé : {row['test_propose']}")
        print(f"Variables : {row['variable_1']} vs {row['variable_2']}")
        print(f"Justification : {row['justification']}")
        
        rep = input("Voulez-vous exécuter ce test ? (oui/non) : ").strip().lower()
        if rep != "oui":
            continue

        # --- Gestion simple de différents types de tests ---
        if row['test_propose'] in ["t-test", "Mann-Whitney", "ANOVA", "Kruskal-Wallis"]:
            var_num = row['variable_1']
            var_cat = row['variable_2']
            # On suppose que var_cat a 2 ou plus de modalités
            groupes = df.groupby(var_cat)[var_num].apply(list)

            if row['test_propose'] == "t-test":
                stat, p = stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
            elif row['test_propose'] == "Mann-Whitney":
                stat, p = stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
            elif row['test_propose'] == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif row['test_propose'] == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)
            
            print(f"Statistique = {stat:.4f}, p-value = {p:.4g}")

            # Graphique boxplot
            sns.boxplot(x=var_cat, y=var_num, data=df)
            plt.title(f"{row['test_propose']} : {var_num} vs {var_cat}")
            plt.show()

        elif "Corrélation" in row['test_propose']:
            var1 = row['variable_1']
            var2 = row['variable_2']
            if "Pearson" in row['test_propose']:
                corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            else:
                corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
            print(f"Corrélation = {corr:.4f}, p-value = {p:.4g}")
            sns.scatterplot(x=var1, y=var2, data=df)
            plt.title(f"{row['test_propose']} : {var1} vs {var2}")
            plt.show()

        # Pour les lignes multivariées (ex : Régression linéaire / PCA)
        elif "Régression linéaire / PCA" in row['test_propose']:
            vars_num = row['variable_1'].split(", ")
            for var in vars_num:
                rep_var = input(f"Voulez-vous exécuter une analyse pour {var} ? (oui/non) : ").strip().lower()
                if rep_var != "oui":
                    continue
                sns.histplot(df[var].dropna(), kde=True)
                plt.title(f"Distribution de {var}")
                plt.show()
                print(f"Analyse exploratoire terminée pour {var}")
