"""
chatbot.py
==========
Interface terminal du chatbot RAG Puls-Events.

Ce script constitue l'étape 4 du pipeline RAG (mode terminal). Il charge
l'index FAISS construit par build_vector_db.py, récupère les chunks les plus
pertinents pour chaque question utilisateur, puis génère une réponse naturelle
via l'API Mistral.

Architecture RAG (online) :
    Question → Retriever FAISS (MMR k=15) → Filtrage → Prompt → Mistral → Réponse

Usage :
    python scripts/chatbot.py

Prérequis :
    - data/index/faiss_index/ doit exister (lancez build_vector_db.py d'abord)
    - MISTRAL_API_KEY dans le fichier .env
"""

import os
import sys
import re
from datetime import datetime
from dotenv import load_dotenv
import time

# --------------------------------------------------
# 1  Chargement de l'environnement
# --------------------------------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("MISTRAL_API_KEY non trouvee dans le fichier .env")
    sys.exit(1)

# --------------------------------------------------
# 2  Imports
# --------------------------------------------------
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# 3  Embeddings + FAISS
# --------------------------------------------------
#  modèle lu depuis le .env
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH      = "data/index/faiss_index"

print("Chargement embeddings + index FAISS...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

if not os.path.exists(INDEX_PATH):
    print("Index FAISS introuvable : " + INDEX_PATH)
    print("Lancez d'abord : python scripts/build_vector_db.py")
    sys.exit(1)

vector_db = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("Index FAISS charge — " + str(vector_db.index.ntotal) + " vecteurs")

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 40}
)

# --------------------------------------------------
# 4  LLM Mistral
# --------------------------------------------------
llm = ChatMistralAI(
    model="open-mistral-7b",
    api_key=MISTRAL_API_KEY,
    temperature=0.4
)
print("LLM Mistral pret")

# --------------------------------------------------
# 5  Fonctions utilitaires
# --------------------------------------------------

def parse_event(text: str) -> dict | None:
    """
    Extrait les informations structurées d'un chunk de texte RAG.

    Utilise des expressions régulières pour retrouver les champs 'Evenement',
    'Date' et 'Lieu' dans le contenu d'un chunk FAISS. Compatible avec les
    deux formes orthographiques : 'Evenement' (sans accent) et 'Événement'
    (avec accent), car fetch_events.py écrit sans accent dans le champ content.

    Args:
        text (str): Contenu brut d'un chunk FAISS (page_content).

    Returns:
        dict | None: Dictionnaire avec les clés 'name', 'date', 'lieu',
            ou None si les champs obligatoires (name, date) sont absents.

    Example:
        >>> chunk = "Evenement : Concert Jazz. Lieu : Paris. Date : 20/02/2026."
        >>> result = parse_event(chunk)
        >>> print(result)
        {'name': 'Concert Jazz', 'date': '20/02/2026', 'lieu': 'Paris'}
    """
    name_match = re.search(r"[EÉ]v[eé]nement\s*:\s*(.*?)\.", text, re.IGNORECASE)
    date_match = re.search(r"Date\s*:\s*(\d{2}/\d{2}/\d{4})", text)
    lieu_match = re.search(r"Lieu\s*:\s*(.*?)\.", text)

    if not name_match or not date_match:
        return None

    return {
        "name": name_match.group(1).strip(),
        "date": date_match.group(1).strip(),
        "lieu": lieu_match.group(1).strip() if lieu_match else "Paris",
    }


def filter_events(docs: list) -> list:
    """
    Filtre et déduplique les événements récupérés depuis FAISS.

    Applique deux niveaux de filtrage :
        1. Fenêtre temporelle : conserve uniquement les événements dans
           l'intervalle [-1 an, +1 an] par rapport à aujourd'hui.
        2. Déduplication : élimine les doublons basés sur le contenu formaté
           (nom + date + lieu).

    Args:
        docs (list[Document]): Liste de documents LangChain retournés par le retriever.

    Returns:
        list[str]: Liste de chaînes formatées "- nom | Le date | Lieu : lieu",
            sans doublons, dans la fenêtre temporelle valide.

    Example:
        >>> docs = retriever.invoke("concert jazz paris")
        >>> events = filter_events(docs)
        >>> print(events[0])
        - Concert Jazz | Le 20/02/2026 | Lieu : Paris
    """
    now          = datetime.now()
    one_year_ago = now.replace(year=now.year - 1)
    one_year_fut = now.replace(year=now.year + 1)

    seen   = set()
    events = []

    for doc in docs:
        ev = parse_event(doc.page_content)
        if ev is None:
            continue

        try:
            ev_date = datetime.strptime(ev["date"], "%d/%m/%Y")
        except ValueError:
            continue

        if not (one_year_ago.date() <= ev_date.date() <= one_year_fut.date()):
            continue

        line = f"- {ev['name']} | Le {ev['date']} | Lieu : {ev['lieu']}"
        if line not in seen:
            seen.add(line)
            events.append(line)

    return events


def build_prompt(question: str, events: list) -> str:
    """
    Construit le prompt envoyé au modèle Mistral.

    Assemble un prompt structuré au format [INST]...[/INST] compatbile avec
    Mistral-7B-Instruct. Le contexte (liste des événements filtrés) est injecté
    directement dans le prompt pour garantir que le modèle répond uniquement
    à partir des données indexées.

    Args:
        question (str): Question posée par l'utilisateur.
        events (list[str]): Liste des événements filtrés (max 5 recommandés).

    Returns:
        str: Prompt complet formaté pour l'API Mistral.

    Example:
        >>> prompt = build_prompt("concerts à Paris ?", ["- Jazz | 20/02 | Paris"])
        >>> print(prompt[:80])
        [INST] Tu es l'assistant Puls-Events...
    """
    contexte = "\n".join(events[:5])
    return (
        "[INST] Tu es l'assistant Puls-Events. "
        "En utilisant UNIQUEMENT la liste suivante, reponds a l'utilisateur "
        "de facon polie, naturelle et en francais.\n\n"
        "LISTE DES EVENEMENTS :\n"
        + contexte
        + "\n\nQUESTION : " + question + " [/INST]"
    )


# --------------------------------------------------
# 6  Boucle de chat
# --------------------------------------------------
print("""
==========================================
    Chatbot Puls-Events (Mistral API)
    Tape 'exit' pour quitter
==========================================
""")

while True:
    try:
        question = input("Vous : ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAu revoir !")
        break

    if not question:
        continue
    if question.lower() in {"exit", "quit"}:
        print("Au revoir !")
        break

    print("Recherche en cours...\n")

    try:
        start_total = time.time()

        t0 = time.time()
        docs = retriever.invoke(question)
        t1 = time.time()

        events = filter_events(docs)
        t2 = time.time()

        if not events:
            print("Désolé, aucun événement trouvé.\n")
            continue

        prompt = build_prompt(question, events)
        response = llm.invoke(prompt)
        t3 = time.time()

        # ✅ Réponse
        print("Assistant :\n" + response.content + "\n")

        # ✅ Temps total
        total = round(t3 - start_total, 2)
        print(f"⏱️ Temps total : {total} secondes")

        # ✅ Détails (BONUS)
        print(f"   🔎 Retrieval : {round(t1 - t0, 2)} s")
        print(f"   🧹 Filtrage  : {round(t2 - t1, 2)} s")
        print(f"   🤖 LLM       : {round(t3 - t2, 2)} s\n")

    except Exception as e:
        print("Erreur : " + str(e) + "\n")