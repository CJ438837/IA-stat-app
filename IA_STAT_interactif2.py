import streamlit as st
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
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

# --- Fonction interactive complète ---
def propose_tests_interactif(types_df, distribution_df, df, mots_cles):
    num_vars = types_df[types_df['type']=="numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    st.header("🧮 Tests statistiques interactifs")

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
            key_test = f"{num}_{cat}_test"
            if key_test not in st.session_state:
                st.session_state[key_test] = test_options[0]
            test_name = st.selectbox("Choisir le test :", test_options, index=test_options.index(st.session_state[key_test]), key=key_test)
            st.session_state[key_test] = test_name

            # Appariement
            apparie = False
            key_app = f"{num}_{cat}_apparie"
            if test_name in ["t-test","Mann-Whitney"]:
                if key_app not in st.session_state:
                    st.session_state[key_app] = False
                apparie = st.radio("Données appariées ?", [False, True], index=int(st.session_state[key_app]), key=key_app)
                st.session_state[key_app] = apparie

            # PubMed
            liens = rechercher_pubmed_test(test_name, mots_cles)
            if liens:
                st.markdown("**Articles PubMed suggérés :**")
                for lien in liens:
                    st.markdown(f"- [{lien}]({lien})")

            # Exécution test
            key_exec = f"{num}_{cat}_exec"
            if key_exec not in st.session_state:
                st.session_state[key_exec] = None

            if st.button(f"Exécuter le test {test_name}", key=f"btn_{key_exec}"):
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
                    st.session_state[key_exec] = (stat, p)
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.session_state[key_exec] = None

            # Affichage résultats si déjà calculés
            if st.session_state[key_exec] is not None:
                stat, p = st.session_state[key_exec]
                st.write(f"Statistique = {stat:.4f}, p-value = {p:.4g}")
                st.write("→ Impact significatif" if p<0.05 else "→ Pas d'impact significatif")

                fig, ax = plt.subplots()
                sns.boxplot(x=cat, y=num, data=df, ax=ax)
                ax.set_title(f"{test_name} : {num} vs {cat}")
                st.pyplot(fig)

    # --- 2️⃣ Corrélations numériques ---
    st.subheader("2️⃣ Corrélations numériques")
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        with st.expander(f"Corrélation : {var1} vs {var2}"):
            key_corr = f"corr_{var1}_{var2}"
            if key_corr not in st.session_state:
                st.session_state[key_corr] = None

            if st.button(f"Exécuter la corrélation {var1} vs {var2}", key=f"btn_{key_corr}"):
                corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna()) if test_type=="Pearson" else stats.spearmanr(df[var1].dropna(), df[var2].dropna())
                st.session_state[key_corr] = (corr, p)

            if st.session_state[key_corr] is not None:
                corr, p = st.session_state[key_corr]
                st.write(f"Corrélation = {corr:.4f}, p-value = {p:.4g}")
                st.write("→ Corrélation significative" if p<0.05 else "→ Pas de corrélation significative")

                fig, ax = plt.subplots()
                sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
                ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")
                st.pyplot(fig)

    # --- 3️⃣ Variables catégorielles ---
    st.subheader("3️⃣ Variables catégorielles")
    for var1, var2 in itertools.combinations(cat_vars, 2):
        with st.expander(f"{var1} vs {var2}"):
            key_cat = f"cat_{var1}_{var2}"
            if key_cat not in st.session_state:
                st.session_state[key_cat] = None

            liens = rechercher_pubmed_test("Chi-square test", mots_cles)
            if liens:
                st.markdown("**Articles PubMed :**")
                for lien in liens:
                    st.markdown(f"- [{lien}]({lien})")

            if st.button(f"Exécuter test catégoriel {var1} vs {var2}", key=f"btn_{key_cat}"):
                contingency_table = pd.crosstab(df[var1], df[var2])
                try:
                    if contingency_table.size <= 4:
                        stat, p = stats.fisher_exact(contingency_table)
                        test_name = "Fisher exact"
                    else:
                        stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                        test_name = "Chi²"
                    st.session_state[key_cat] = (stat, p, test_name)
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.session_state[key_cat] = None

            if st.session_state[key_cat] is not None:
                stat, p, test_name = st.session_state[key_cat]
                st.write(f"{test_name} : statistique={stat:.4f}, p-value={p:.4g}")
                st.write("→ Dépendance significative" if p<0.05 else "→ Pas de dépendance significative")

                fig, ax = plt.subplots()
                sns.heatmap(pd.crosstab(df[var1], df[var2]), annot=True, fmt="d", cmap="coolwarm", ax=ax)
                ax.set_title(f"{test_name} : {var1} vs {var2}")
                st.pyplot(fig)

    # --- 4️⃣ Régression linéaire multiple ---
    if len(num_vars) > 1:
        st.subheader("4️⃣ Régression linéaire multiple")
        if "run_regression" not in st.session_state:
            st.session_state["run_regression"] = False
        run_regression = st.checkbox("Exécuter régression linéaire multiple", value=st.session_state["run_regression"])
        st.session_state["run_regression"] = run_regression

        if run_regression:
            X = df[num_vars].dropna()
            cible = st.selectbox("Variable dépendante :", num_vars)
            y = X[cible]
            X_pred = X.drop(columns=[cible])
            model = LinearRegression()
            model.fit(X_pred, y)
            y_pred = model.predict(X_pred)
            residus = y - y_pred

            st.write(f"R² = {model.score(X_pred, y):.4f}")
            stat, p = stats.shapiro(residus)
            st.write(f"Shapiro-Wilk résidus : stat={stat:.4f}, p={p:.4g}")
            st.write("Résidus normalement distribués" if p>0.05 else "⚠️ Résidus non normaux")

            coef_df = pd.DataFrame({"Variable": X_pred.columns, "Coefficient": model.coef_})
            st.table(coef_df)
            st.write(f"Intercept : {model.intercept_:.4f}")

            fig, axes = plt.subplots(2,2, figsize=(12,10))
            sns.scatterplot(x=y_pred, y=residus, ax=axes[0,0])
            axes[0,0].axhline(0, color='red', linestyle='--')
            axes[0,0].set_title("Résidus vs Prédit")
            sns.histplot(residus, kde=True, ax=axes[0,1], color='skyblue')
            axes[0,1].set_title("Distribution résidus")
            stats.probplot(residus, dist="norm", plot=axes[1,0])
            axes[1,0].set_title("QQ-Plot résidus")
            sns.scatterplot(x=y, y=y_pred, ax=axes[1,1])
            axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
            axes[1,1].set_title("Observé vs Prédit")
            plt.tight_layout()
            st.pyplot(fig)

    # --- 5️⃣ PCA ---
    if len(num_vars) > 1:
        st.subheader("5️⃣ Analyse en Composantes Principales (PCA)")
        if "run_pca" not in st.session_state:
            st.session_state["run_pca"] = False
        run_pca = st.checkbox("Exécuter PCA", value=st.session_state["run_pca"])
        st.session_state["run_pca"] = run_pca

        if run_pca:
            X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_
            cum_var = explained_variance.cumsum()
            n_comp = (cum_var<0.8).sum()+1
            st.write(f"{n_comp} composantes expliquent ~80% de la variance")
            loading_matrix = pd.DataFrame(pca.components_.T, index=num_vars,
                                          columns=[f"PC{i+1}" for i in range(len(num_vars))])
            st.write(loading_matrix.iloc[:,:n_comp])

            fig, ax = plt.subplots()
            ax.scatter(components[:,0], components[:,1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Projection individus PC1 vs PC2")
            st.pyplot(fig)

    # --- 6️⃣ MCA pour variables catégorielles ---
    if len(cat_vars) > 1:
        st.subheader("6️⃣ Analyse des Correspondances Multiples (MCA)")
        if "run_mca" not in st.session_state:
            st.session_state["run_mca"] = False
        run_mca = st.checkbox("Exécuter MCA", value=st.session_state["run_mca"])
        st.session_state["run_mca"] = run_mca

        if run_mca:
            try:
                import prince
                df_cat = df[cat_vars].fillna("Missing")
                mca = prince.MCA(n_components=2, random_state=42)
                mca = mca.fit(df_cat)

                var_expl = mca.explained_inertia_ if hasattr(mca,"explained_inertia_") else mca.explained_variance_ratio_
                st.write(f"Variance expliquée : {var_expl[0]*100:.2f}%, {var_expl[1]*100:.2f}%")
                coords = mca.column_coordinates(df_cat)
                ind_coords = mca.row_coordinates(df_cat)

                fig, ax = plt.subplots()
                ax.scatter(ind_coords[0], ind_coords[1], alpha=0.6)
                ax.set_xlabel("Dim 1")
                ax.set_ylabel("Dim 2")
                ax.set_title("Projection individus MCA")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.scatter(coords[0], coords[1], color='red', alpha=0.7)
                for i, label in enumerate(coords.index):
                    ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=9, color='darkred')
                ax.set_xlabel("Dim 1")
                ax.set_ylabel("Dim 2")
                ax.set_title("Projection catégories MCA")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(6,6))
                circle = plt.Circle((0,0),1, color='gray', fill=False)
                ax.add_artist(circle)
                for i, label in enumerate(coords.index):
                    ax.arrow(0,0, coords.iloc[i,0], coords.iloc[i,1], color='blue', alpha=0.5, head_width=0.03)
                    ax.text(coords.iloc[i,0]*1.1, coords.iloc[i,1]*1.1, label, color='blue', ha='center', va='center', fontsize=8)
                ax.set_xlim(-1.1,1.1)
                ax.set_ylim(-1.1,1.1)
                ax.axhline(0,color='gray',lw=0.5)
                ax.axvline(0,color='gray',lw=0.5)
                ax.set_title("Cercle des corrélations (MCA)")
                st.pyplot(fig)
            except ImportError:
                st.warning("⚠️ Module 'prince' non installé. Exécutez : pip install prince")
            except Exception as e:
                st.error(f"Erreur MCA : {e}")

    # --- 7️⃣ Régression logistique pour variables binaires ---
    st.subheader("7️⃣ Régression logistique pour variables binaires")
    for cat in cat_vars:
        if df[cat].dropna().nunique()==2:
            key_log = f"logistic_{cat}"
            if key_log not in st.session_state:
                st.session_state[key_log] = False
            run_log = st.checkbox(f"Exécuter régression logistique : {cat}", value=st.session_state[key_log])
            st.session_state[key_log] = run_log

            if run_log:
                X = df[num_vars].dropna()
                y = df[cat].loc[X.index]
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                st.write("Coefficients :", dict(zip(num_vars, model.coef_[0])))
                st.write(f"Intercept : {model.intercept_[0]}")

    st.success("✅ Tous les tests interactifs terminés")
