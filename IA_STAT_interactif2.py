import streamlit as st

def propose_tests_interactif(types_df, distribution_df, df, keywords):
    tests_selectionnes = []

    st.markdown("### 💡 Tests proposés automatiquement")
    for _, row in types_df.iterrows():
        var = row["variable"]
        var_type = row["type"]
        st.write(f"**{var}** ({var_type})")

    st.divider()
    st.subheader("Sélection manuelle des tests")

    col1, col2 = st.columns(2)
    with col1:
        test_ttest = st.checkbox("T-test")
        test_anova = st.checkbox("ANOVA")
    with col2:
        test_chi2 = st.checkbox("Chi²")
        test_mannwhitney = st.checkbox("Mann-Whitney")

    if test_ttest:
        tests_selectionnes.append({"nom": "T-test", "type": "t-test", "variables": ("Age", "Poids")})
    if test_anova:
        tests_selectionnes.append({"nom": "ANOVA", "type": "anova", "variables": ("Age", "Poids")})
    if test_chi2:
        tests_selectionnes.append({"nom": "Chi²", "type": "chi2", "variables": ("Groupe", "Age")})
    if test_mannwhitney:
        tests_selectionnes.append({"nom": "Mann-Whitney", "type": "mannwhitney", "variables": ("Age", "Poids")})

    return tests_selectionnes
