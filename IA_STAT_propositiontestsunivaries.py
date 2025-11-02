import pandas as pd
import itertools

def propose_tests(types_df, distribution_df, df):
    """
    Propose automatiquement des tests statistiques selon le type et la distribution des variables.

    Args:
        types_df (DataFrame) : tableau des types détectés
        distribution_df (DataFrame) : tableau avec verdict normal / non normal
        df (DataFrame) : le DataFrame complet des données

    Returns:
        DataFrame : tableau des tests proposés avec justification
    """
    tests = []

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    # --- 1️⃣ Numérique vs Catégoriel ---
    for num, cat in itertools.product(num_vars, cat_vars):
        # Calcul du nombre de modalités directement à partir du DataFrame
        n_modalites = df[cat].dropna().nunique()

        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

        if n_modalites == 2:
            if verdict == "Normal":
                test = "t-test"
                justification = "Numérique normal vs Catégoriel à 2 modalités"
            else:
                test = "Mann-Whitney"
                justification = "Numérique non normal vs Catégoriel à 2 modalités"
        elif n_modalites > 2:
            if verdict == "Normal":
                test = "ANOVA"
                justification = "Numérique normal vs Catégoriel >2 modalités"
            else:
                test = "Kruskal-Wallis"
                justification = "Numérique non normal vs Catégoriel >2 modalités"
        else:
            test = "unknown"
            justification = "Impossible de déterminer le test"

        tests.append({
            "variable_1": num,
            "variable_2": cat,
            "test_propose": test,
            "justification": justification
        })

    # --- 2️⃣ Deux variables continues ---
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"
        justification = f"Corrélation entre deux numériques ({verdict1}, {verdict2})"
        tests.append({
            "variable_1": var1,
            "variable_2": var2,
            "test_propose": f"Corrélation ({test_type})",
            "justification": justification
        })

    # --- 3️⃣ Deux variables catégorielles ---
    for var1, var2 in itertools.combinations(cat_vars, 2):
        test = "Khi² / Fisher"
        justification = "Deux variables catégorielles"
        tests.append({
            "variable_1": var1,
            "variable_2": var2,
            "test_propose": test,
            "justification": justification
        })

    # --- 4️⃣ Plusieurs variables numériques ---
    if len(num_vars) > 1:
        tests.append({
            "variable_1": ", ".join(num_vars),
            "variable_2": "dépendante si existante",
            "test_propose": "Régression linéaire / PCA",
            "justification": "Plusieurs variables numériques"
        })

    # --- 5️⃣ Variable binaire dépendante ---
    for cat in cat_vars:
        n_modalites = df[cat].dropna().nunique()
        if n_modalites == 2:
            tests.append({
                "variable_1": ", ".join(num_vars),
                "variable_2": cat,
                "test_propose": "Régression logistique",
                "justification": "Variable dépendante binaire"
            })

    return pd.DataFrame(tests)
