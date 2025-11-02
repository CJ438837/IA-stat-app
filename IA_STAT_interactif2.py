import streamlit as st
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Exemple de fonction PubMed simplifiée
def rechercher_pubmed_test(test_name, mots_cles):
    return [f"https://pubmed.ncbi.nlm.nih.gov/{i}/" for i in range(1,4)]  # mock

def propose_tests_interactif(df, types_df, distribution_df, mots_cles):
    num_vars = types_df[types_df['type']=="numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    st.header("🧮 Tests statistiques interactifs")

    if 'tests_executed' not in st.session_state:
        st.session_state['tests_executed'] = []

    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

        if n_modalites == 2:
            test_options = ["t-test" if verdict=="Normal" else "Mann-Whitney"]
        elif n_modalites > 2:
            test_options = ["ANOVA" if verdict=="Normal" else "Kruskal-Wallis"]
        else:
            test_options = ["unknown"]

        with st.expander(f"{num} vs {cat}", expanded=False):
            form_key = f"form_{num}_{cat}"
            with st.form(key=form_key):
                test_name = st.selectbox("Choisir le test :", test_options)
                apparie = False
                if test_name in ["t-test","Mann-Whitney"]:
                    apparie = st.radio("Données appariées ?", [False, True])

                submit = st.form_submit_button("Ajouter et exécuter ce test")
                
                if submit:
                    # Execution du test
                    groupes = df.groupby(cat)[num].apply(list)
                    try:
                        if test_name == "t-test":
                            stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
                        elif test_name == "Mann-Whitney":
                            stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
                        elif test_name == "ANOVA":
                            stat, p = stats.f_oneway(*groupes)
                        elif test_name == "Kruskal-Wallis":
                            stat, p = stats.kruskal(*groupes)
                        else:
                            stat, p = None, None

                        st.session_state['tests_executed'].append({
                            'num': num,
                            'cat': cat,
                            'test': test_name,
                            'apparie': apparie,
                            'stat': stat,
                            'p': p
                        })

                        if stat is not None:
                            st.write(f"**Résultat : {test_name}** — Stat = {stat:.4f}, p-value = {p:.4g}")
                            st.write("→ Impact significatif" if p<0.05 else "→ Pas d'impact significatif")

                        fig, ax = plt.subplots()
                        sns.boxplot(x=cat, y=num, data=df, ax=ax)
                        ax.set_title(f"{test_name} : {num} vs {cat}")
                        st.pyplot(fig)

                        # PubMed
                        liens = rechercher_pubmed_test(test_name, mots_cles)
                        st.markdown("**Articles PubMed suggérés :**")
                        for lien in liens:
                            st.markdown(f"- [{lien}]({lien})")

                    except Exception as e:
                        st.error(f"Erreur : {e}")

    # Afficher les tests déjà exécutés
    if st.session_state['tests_executed']:
        st.markdown("### ✅ Tests exécutés récemment")
        for t in st.session_state['tests_executed']:
            st.write(f"{t['num']} vs {t['cat']} : {t['test']} — Stat={t['stat']:.4f}, p={t['p']:.4g}")
