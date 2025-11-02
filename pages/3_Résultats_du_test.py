import streamlit as st

st.title("📈 Résultats des tests")

if "tests_selectionnes" in st.session_state:
    tests = st.session_state["tests_selectionnes"]
    if not tests:
        st.info("Aucun test sélectionné.")
    else:
        for test in tests:
            st.write(f"**{test['nom']}** sur les variables {test['variables']}")
            # Ici tu peux ajouter le calcul réel si tu veux l’exécuter automatiquement
else:
    st.info("Aucun test sélectionné pour le moment.")
