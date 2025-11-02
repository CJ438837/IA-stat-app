import streamlit as st
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from Bio import Entrez

# --- Fonction PubMed ---
def rechercher_pubmed_test(test_name, mots_cles, email="votre.email@example.com", max_results=3):
    Entrez.email = email
    query = f"{test_name} AND (" + " OR ".join(mots_cles) + ")"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    pmids = record['IdList']
    liens = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]
    return liens

# --- Initialisation session_state ---
if 'tests_a_executer' not in st.session_state:
    st.session_state['tests_a_executer'] = []

# --- Fonction interactive ---
def propose_tests_interactif(types_df, distribution_df, df, mots_cles):
    num_vars = types_df[types_df['type']=="numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    st.header("🧮 Sélection des tests statistiques")

    # --- 1️⃣ Numérique vs Catégoriel ---
    st.subheader("1️⃣ Numérique vs Catégoriel")
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

        if n_modalites == 2:
            test_options = ["t-test" if verdict=="Normal" else "Mann-Whitney"]
        elif n_modalites > 2:
            test_options = ["ANOVA" if verdict=="Normal" else "Kruskal-Wallis"]
        else:
            test_options = ["unknown"]

        with st.expander(f"{num} vs {cat}"):
            with st.form(key=f"form_{num}_{cat}"):
                test_name = st.selectbox("Choisir le test :", test_options, key=f"select_{num}_{cat}")
                apparie = False
                if test_name in ["t-test","Mann-Whitney"]:
                    apparie = st.radio("Données appariées ?", [False, True], index=0, key=f"radio_{num}_{cat}")

                if st.form_submit_button("Ajouter ce test"):
                    st.session_state['tests_a_executer'].append({
                        'type': 'num_vs_cat',
                        'num': num,
                        'cat': cat,
                        'test': test_name,
                        'apparie': apparie
                    })
                    st.success(f"Test {test_name} ajouté à la liste")

    # --- Affichage des tests sélectionnés ---
    if st.session_state['tests_a_executer']:
        st.markdown("### 📝 Tests sélectionnés")
        for i, t in enumerate(st.session_state['tests_a_executer']):
            st.write(f"{i+1}. {t}")

        if st.button("▶️ Exécuter tous les tests sélectionnés"):
            for t in st.session_state['tests_a_executer']:
                if t['type']=='num_vs_cat':
                    groupes = df.groupby(t['cat'])[t['num']].apply(list)
                    try:
                        if t['test'] == "t-test":
                            stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1]) if t['apparie'] else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
                        elif t['test'] == "Mann-Whitney":
                            stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1]) if t['apparie'] else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
                        elif t['test'] == "ANOVA":
                            stat, p = stats.f_oneway(*groupes)
                        elif t['test'] == "Kruskal-Wallis":
                            stat, p = stats.kruskal(*groupes)
                        else:
                            stat, p = None, None

                        st.write(f"**{t['test']} : {t['num']} vs {t['cat']}**")
                        if stat is not None:
                            st.write(f"Statistique = {stat:.4f}, p-value = {p:.4g}")
                            st.write("→ Impact significatif" if p<0.05 else "→ Pas d'impact significatif")

                        fig, ax = plt.subplots()
                        sns.boxplot(x=t['cat'], y=t['num'], data=df, ax=ax)
                        ax.set_title(f"{t['test']} : {t['num']} vs {t['cat']}")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur : {e}")

            # Nettoyage après exécution
            st.session_state['tests_a_executer'] = []
            st.success("✅ Tous les tests exécutés")

