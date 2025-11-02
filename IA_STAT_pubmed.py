import streamlit as st
import re
from Bio import Entrez

# --- Config PubMed ---
Entrez.email = "ton.email@example.com"

st.title("🔍 Recherche PubMed depuis une description d'étude")

# --- Entrée utilisateur ---
description = st.text_area("Décrivez votre étude en quelques phrases :")

if description:
    # --- Extraction des mots alphabétiques ---
    tokens = re.findall(r'\b\w+\b', description.lower())
    stopwords_fr = set([
        "le","la","les","un","une","des","de","du","et","en","au","aux","avec",
        "pour","sur","dans","par","au","a","ce","ces","est","sont","ou","où","se",
        "sa","son","que","qui","ne","pas","plus","moins","comme","donc"
    ])
    keywords_fr = [w for w in tokens if w not in stopwords_fr]

    if keywords_fr:
        query = " OR ".join(keywords_fr)
        st.markdown(f"**Mots-clés français extraits :** {', '.join(keywords_fr)}")
        st.markdown(f"**Requête PubMed générée :** {query}")

        # --- Recherche PubMed ---
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        pmids = record['IdList']
        if not pmids:
            st.warning("⚠️ Aucun article trouvé.")
        else:
            st.success(f"✅ {len(pmids)} articles trouvés :")
            for i, pmid in enumerate(pmids, 1):
                st.markdown(f"{i}. [https://pubmed.ncbi.nlm.nih.gov/{pmid}/](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
    else:
        st.warning("⚠️ Aucun mot-clé extrait de la description.")
