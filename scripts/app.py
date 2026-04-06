"""
app.py
======
Interface web Streamlit du chatbot RAG Puls-Events.

Ce script est la version interface graphique de chatbot.py. Il expose le même
pipeline RAG (FAISS + Mistral) via une interface Streamlit accessible dans
le navigateur. Il intègre un filtrage intelligent par mois/année détecté
automatiquement dans la question de l'utilisateur.

Fonctionnalités :
    - Champ de saisie + bouton "Chercher"
    - Détection automatique du mois et de l'année dans la question
    - Recherche sémantique FAISS (MMR) ou scan complet si date détectée
    - Filtrage par date, déduplication des résultats
    - Génération de réponse via API Mistral
    - Affichage des sources dans un expander

Usage :
    streamlit run scripts/app.py

Prérequis :
    - data/index/faiss_index/ doit exister (lancez build_vector_db.py d'abord)
    - MISTRAL_API_KEY dans le fichier .env
    - pip install streamlit
"""

import os
import re
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
# --------------------------------------------------
# 1  Environnement
# --------------------------------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY manquante dans le fichier .env")
    st.stop()

# --------------------------------------------------
# 2  UI
# --------------------------------------------------
st.set_page_config(page_title="Puls-Events", page_icon="🎉", layout="centered")
st.title("🎉 Chatbot Puls-Events")
st.caption("Posez vos questions sur les événements culturels parisiens")

# --------------------------------------------------
# 3  FAISS (mis en cache)
# --------------------------------------------------
@st.cache_resource
def load_db() -> FAISS:
    """
    Charge l'index FAISS et le modèle d'embedding, mis en cache par Streamlit.

    Utilise @st.cache_resource pour ne charger l'index qu'une seule fois
    au démarrage de l'application, même si l'utilisateur interagit plusieurs fois.
    Le modèle d'embedding doit être identique à celui utilisé lors de la création
    de l'index (all-MiniLM-L6-v2, dim=384, normalisé).

    Returns:
        FAISS: Base vectorielle chargée depuis data/index/faiss_index/.

    Raises:
        Exception: Si le dossier d'index est absent ou corrompu.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        "data/index/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm() -> ChatMistralAI:
    """
    Instancie le modèle Mistral via l'API officielle, mis en cache par Streamlit.

    Utilise @st.cache_resource pour créer le client LLM une seule fois.
    La température basse (0.4) favorise des réponses factuelles et cohérentes
    tout en conservant un style naturel.

    Returns:
        ChatMistralAI: Client LLM configuré pour Mistral-7B-Instruct.
    """
    return ChatMistralAI(
        model="open-mistral-7b",
        api_key=MISTRAL_API_KEY,
        temperature=0.4
    )


vector_db = load_db()
llm       = load_llm()

# --------------------------------------------------
# 4  Détection du mois/année dans la question
# --------------------------------------------------

MOIS_MAP = {
    "janvier": "01", "fevrier": "02", "février": "02",
    "mars": "03",    "avril": "04",   "mai": "05",
    "juin": "06",    "juillet": "07", "aout": "08", "août": "08",
    "septembre": "09", "octobre": "10", "novembre": "11",
    "decembre": "12", "décembre": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "june": "06",    "july": "07",     "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def detect_date_filter(question: str) -> tuple:
    """
    Détecte un mois et/ou une année dans la question de l'utilisateur.

    Parcourt le dictionnaire MOIS_MAP pour identifier un nom de mois (français
    ou anglais) dans la question, et utilise une regex pour extraire une année
    au format 202x. Ces informations sont utilisées pour filtrer les résultats
    FAISS par date de manière précise.

    Args:
        question (str): Question posée par l'utilisateur en langage naturel.

    Returns:
        tuple[str | None, str | None]: Tuple (mois_num, annee_str) où mois_num
            est le numéro de mois sur 2 chiffres (ex: "03") et annee_str est
            l'année sur 4 chiffres (ex: "2026"). Retourne (None, None) si
            aucune date n'est détectée.

    Example:
        >>> detect_date_filter("événements à Paris en mars 2026")
        ('03', '2026')
        >>> detect_date_filter("concerts de jazz")
        (None, None)
    """
    q           = question.lower()
    mois_found  = None
    annee_found = None

    for mot, num in MOIS_MAP.items():
        if mot in q:
            mois_found = num
            break

    annee_match = re.search(r"\b(202[0-9])\b", q)
    if annee_match:
        annee_found = annee_match.group(1)

    return mois_found, annee_found


# --------------------------------------------------
# 5  Parsing et filtrage des événements
# --------------------------------------------------

def parse_event(text: str) -> dict | None:
    """
    Extrait les informations structurées d'un chunk de texte FAISS.

    Applique trois expressions régulières pour retrouver le nom de l'événement,
    sa date et son lieu depuis le contenu brut d'un chunk. La regex pour le
    nom est insensible aux accents pour gérer les deux formes : 'Evenement'
    (écrit par fetch_events.py) et 'Événement' (forme accentuée).

    Args:
        text (str): Contenu brut d'un chunk FAISS (page_content).

    Returns:
        dict | None: Dictionnaire avec les clés 'name', 'date', 'lieu',
            ou None si les champs obligatoires sont absents du chunk.

    Example:
        >>> chunk = "Evenement : Jazz Café. Lieu : 10 rue de la Paix, Paris. Date : 15/02/2026."
        >>> parse_event(chunk)
        {'name': 'Jazz Café', 'date': '15/02/2026', 'lieu': '10 rue de la Paix, Paris'}
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


def filter_events(docs: list, mois_filter: str = None, annee_filter: str = None) -> list:
    """
    Filtre les événements selon la fenêtre temporelle et les critères de date détectés.

    Applique trois niveaux de filtrage successifs :
        1. Fenêtre globale [-1 an, +1 an] : élimine les événements trop anciens ou trop futurs.
        2. Filtre mois (optionnel) : conserve uniquement le mois détecté dans la question.
        3. Filtre année (optionnel) : conserve uniquement l'année détectée dans la question.
    Puis déduplique les résultats pour éviter les doublons dans la réponse.

    Args:
        docs (list[Document]): Documents LangChain retournés par FAISS.
        mois_filter (str | None): Numéro de mois sur 2 chiffres (ex: "03"), ou None.
        annee_filter (str | None): Année sur 4 chiffres (ex: "2026"), ou None.

    Returns:
        list[str]: Liste de chaînes formatées "nom - date - lieu", sans doublons,
            correspondant aux critères de filtrage.

    Example:
        >>> docs = vector_db.similarity_search("jazz", k=50)
        >>> events = filter_events(docs, mois_filter="02", annee_filter="2026")
        >>> print(events[0])
        Concert Jazz - 20/02/2026 - 142 Avenue de Flandre, Paris
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
        if mois_filter and f"{ev_date.month:02d}" != mois_filter:
            continue
        if annee_filter and str(ev_date.year) != annee_filter:
            continue

        line = f"{ev['name']} - {ev['date']} - {ev['lieu']}"
        if line not in seen:
            seen.add(line)
            events.append(line)

    return events


# --------------------------------------------------
# 6  Interface utilisateur
# --------------------------------------------------
question = st.text_input(
    "Pose ta question :",
    placeholder="Ex : événements à Paris en octobre 2025...",
)

clicked = st.button("🔍 Chercher")

# --------------------------------------------------
# 7  Pipeline RAG — Recherche et génération
# --------------------------------------------------
if clicked and not question:
    st.warning("Veuillez saisir une question.")

if question and clicked:
    start_total = time.time()

    mois_filter, annee_filter = detect_date_filter(question)

    with st.spinner("Recherche en cours..."):
        t0 = time.time()

        if mois_filter or annee_filter:
            all_docs = vector_db.similarity_search(
                question,
                k=vector_db.index.ntotal
            )
        else:
            retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "fetch_k": 40}
            )
            all_docs = retriever.invoke(question)

        t1 = time.time()

        events = filter_events(all_docs, mois_filter, annee_filter)
        t2 = time.time()

    if not events:
        st.warning("Aucun événement trouvé pour cette recherche.")
    else:
        contexte = "\n".join(events[:8])

        prompt = (
            "[INST] Tu es l'assistant Puls-Events. "
            "En utilisant UNIQUEMENT la liste suivante, reponds en francais "
            "de facon naturelle et polie.\n\n"
            "LISTE DES EVENEMENTS :\n"
            + contexte
            + "\n\nQUESTION : " + question + " [/INST]"
        )

        response = llm.invoke(prompt)
        t3 = time.time()

        # ✅ Réponse
        st.success(response.content)

        # ✅ Temps global
        total_time = round(t3 - start_total, 2)
        st.info(f"⏱️ Temps total : {total_time} secondes")

        # ✅ Détail (bonus très pro)
        with st.expander("⚙️ Détails des performances"):
            st.write(f"🔎 Retrieval : {round(t1 - t0, 2)} s")
            st.write(f"🧹 Filtrage  : {round(t2 - t1, 2)} s")
            st.write(f"🤖 LLM       : {round(t3 - t2, 2)} s")

        # ✅ Sources
        with st.expander(f"Voir les événements sources ({len(events)} trouvés)"):
            for ev in events:
                st.write("• " + ev)
