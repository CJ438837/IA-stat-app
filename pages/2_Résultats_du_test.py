import streamlit as st
import pandas as pd
from scipy import stats

st.title("📈 Résultats des tests statistiques")

if "tests_selectionnes" not in st.session_state or not st.session_state["tests_selectionnes"]:
    st.warning("⚠️ Aucun test sélectionné. Retournez à la page précédente.")
    st.stop()

df = st.session_state["df"]
tests_selectionnes = st.session_state["tests_selectionnes"]

for test in tests_selectionnes:
    st.subheader(f"🧪 {test['nom']}")
    try:
        var1, var2 = test["variables"]
        if test["type"] == "t-test":
            result = stats.ttest_ind(df[var1], df[var2], equal_var=False)
            st.write(result)
        elif test["type"] == "anova":
            result = stats.f_oneway(df[var1], df[var2])
            st.write(result)
        elif test["type"] == "chi2":
            table = pd.crosstab(df[var1], df[var2])
            chi2, p, dof, expected = stats.chi2_contingency(table)
            st.write(f"Chi²={chi2:.3f}, p={p:.4f}, ddl={dof}")
        else:
            st.info("Test non encore implémenté.")
    except Exception as e:
        st.error(f"Erreur : {e}")

st.divider()
if st.button("⬅️ Retour au choix du test"):
    st.switch_page("1_Choix_du_test.py")
