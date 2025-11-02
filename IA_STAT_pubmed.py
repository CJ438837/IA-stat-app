import re
from googletrans import Translator
from Bio import Entrez

# --- Config PubMed ---
Entrez.email = "ton.email@example.com"

# --- Entrée utilisateur ---
description = input("Décrivez votre étude en quelques phrases : ")

# --- Extraction de mots alphabétiques (évite NLTK) ---
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
query = " OR ".join(keywords_en)

print(f"\nMots-clés français : {keywords_fr}")
print(f"Mots-clés traduits anglais : {keywords_en}")
print(f"Requête PubMed : {query}")

# --- Recherche PubMed (max 10 résultats) ---
handle = Entrez.esearch(db="pubmed", term=query, retmax=10, sort="relevance")
record = Entrez.read(handle)
handle.close()

pmids = record['IdList']
if not pmids:
    print("\nAucun article trouvé.")
else:
    print("\n=== Articles PubMed trouvés ===")
    for i, pmid in enumerate(pmids, 1):
        print(f"{i}. https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
