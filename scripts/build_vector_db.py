"""
build_vector_db.py
==================
Pipeline de vectorisation et d'indexation FAISS pour le système RAG Puls-Events.

Ce script constitue l'étape 3 du pipeline RAG. Il charge le dataset nettoyé
produit par fetch_events.py, découpe les textes en chunks, les encode en vecteurs
sémantiques via un modèle d'embedding, puis les indexe dans une base FAISS.

Pipeline :
    events_clean.csv → Chunking → Embedding (MiniLM) → Index FAISS

Étapes :
    0. Chargement des variables d'environnement (.env)
    1. Chargement du dataset CSV
    2. Découpage des textes en chunks (RecursiveCharacterTextSplitter)
    3. Chargement du modèle d'embedding (all-MiniLM-L6-v2)
    4. Création de l'index FAISS
    5. Sauvegarde de l'index sur disque
    6. Tests de recherche sémantique post-build

Usage :
    python scripts/build_vector_db.py

Prérequis :
    - data/events_clean.csv doit exister (lancez fetch_events.py d'abord)
    - HF_TOKEN optionnel dans .env (améliore le rate limit HuggingFace)

Sortie :
    data/index/faiss_index/index.faiss
    data/index/faiss_index/index.pkl
"""

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --------------------------------------------------
# 0  Chargement des variables d'environnement
# --------------------------------------------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN non défini — téléchargements non authentifiés (rate limit réduit)")
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    print("HuggingFace token chargé avec succès")

# --------------------------------------------------
# 1  Chargement du dataset
# --------------------------------------------------
DATA_PATH = "data/events_clean.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
total_events = len(df)
print(f"Nombre d'événements chargés : {total_events}")

df = df.dropna(subset=["content"])
events_after_clean = len(df)
dropped = total_events - events_after_clean
if dropped > 0:
    print(f"{dropped} événement(s) supprimé(s) (contenu vide)")

REQUIRED_METADATA_COLS = ["title", "city", "start_date"]
OPTIONAL_METADATA_COLS = ["category", "price", "url"]

for col in REQUIRED_METADATA_COLS:
    if col not in df.columns:
        print(f"Colonne manquante : '{col}' — métadonnée remplacée par 'N/A'")
        df[col] = "N/A"

# --------------------------------------------------
# 2  Chunking
# --------------------------------------------------
def build_documents(dataframe: pd.DataFrame, splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Découpe les textes de chaque événement en chunks LangChain Document.

    Chaque événement est transformé en plusieurs chunks de 300 tokens maximum
    avec un chevauchement (overlap) de 50 tokens pour préserver le contexte.
    Les métadonnées (title, city, date) sont attachées à chaque chunk pour
    permettre la traçabilité lors de la recherche.

    Args:
        dataframe (pd.DataFrame): Dataset des événements nettoyés.
            Doit contenir les colonnes : content, title, city, start_date.
        splitter (RecursiveCharacterTextSplitter): Splitter LangChain configuré.

    Returns:
        list[Document]: Liste de documents LangChain prêts pour l'indexation.
            Chaque Document contient page_content (chunk) et metadata (title, city, date).

    Example:
        >>> splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        >>> docs = build_documents(df, splitter)
        >>> print(len(docs))  # ~1774 pour 360 événements
    """
    documents = []
    for _, row in dataframe.iterrows():
        chunks = splitter.split_text(str(row["content"]))
        metadata = {
            "title": row.get("title",     "N/A"),
            "city":  row.get("city",      "N/A"),
            "date":  row.get("start_date","N/A"),
        }
        for col in OPTIONAL_METADATA_COLS:
            if col in dataframe.columns and pd.notna(row.get(col)):
                metadata[col] = row[col]
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""],
)

documents = build_documents(df, text_splitter)
nb_chunks = len(documents)

print(f"Nombre de chunks créés : {nb_chunks}")
print(
    f"Taux d'indexation : {events_after_clean}/{total_events} événements "
    f"({events_after_clean / total_events * 100:.1f}%) — "
    f"moyenne {nb_chunks / events_after_clean:.1f} chunks/événement"
)

# --------------------------------------------------
# 3  Chargement du modèle d'embedding
# --------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\nChargement du modèle d'embedding : {EMBEDDING_MODEL} ...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Modèle d'embedding chargé")

# --------------------------------------------------
# 4  Création de l'index FAISS
# --------------------------------------------------
print("\nCréation de l'index FAISS...")
vectorstore = FAISS.from_documents(documents, embeddings)

nb_vectors = vectorstore.index.ntotal
print(f"Index FAISS créé — {nb_vectors} vecteurs indexés")

if nb_vectors != nb_chunks:
    print(f"Divergence : {nb_chunks} chunks créés mais {nb_vectors} vecteurs indexés")

# --------------------------------------------------
# 5  Sauvegarde de l'index
# --------------------------------------------------
INDEX_PATH = "data/index/faiss_index"
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
vectorstore.save_local(INDEX_PATH)
print(f"Index FAISS sauvegardé dans : {INDEX_PATH}")

for f in [os.path.join(INDEX_PATH, "index.faiss"), os.path.join(INDEX_PATH, "index.pkl")]:
    status = "OK" if os.path.exists(f) else "MANQUANT"
    print(f"   [{status}] {f}")

# --------------------------------------------------
# 6  Tests de recherche post-build
# --------------------------------------------------
def run_search_tests(vs: FAISS, queries: list, k: int = 3) -> None:
    """
    Exécute des requêtes de test pour valider la qualité de l'index FAISS.

    Effectue une recherche sémantique pour chaque requête et affiche les
    résultats (titre, ville, date, aperçu du contenu). Permet de vérifier
    visuellement que les résultats sont pertinents et cohérents.

    Args:
        vs (FAISS): Base vectorielle FAISS déjà construite et peuplée.
        queries (list[str]): Liste de requêtes textuelles à tester.
        k (int): Nombre de résultats à retourner par requête. Défaut : 3.

    Returns:
        None: Affiche les résultats dans la console.

    Example:
        >>> run_search_tests(vectorstore, ["concert jazz Paris"], k=3)
        Requête : « concert jazz Paris »
          [1] Stage - Le Jazz ouvre la voix | Paris | 2026-02-03 ...
    """
    print("\nTests de recherche sémantique...\n")
    for query in queries:
        print(f"Requête : « {query} »")
        results = vs.similarity_search(query, k=k)
        for i, doc in enumerate(results, 1):
            title   = doc.metadata.get("title", "N/A")
            city    = doc.metadata.get("city",  "N/A")
            date    = doc.metadata.get("date",  "N/A")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"   [{i}] {title} | {city} | {date}")
            print(f"       → {preview}...")
        print()


run_search_tests(vectorstore, [
    "concert jazz Paris",
    "exposition peinture contemporaine",
    "festival musique été gratuit",
])

print("Pipeline terminé avec succès !")
