# 🎉 Chatbot Puls-Events — Système RAG <br>

Assistant intelligent pour la recommandation d'événements culturels parisiens.  
Basé sur une architecture **RAG** (Retrieval-Augmented Generation) combinant **FAISS**, **LangChain** et **Mistral**.

---

## 📋 Description <br>

Ce projet est un Proof of Concept (POC) développé pour **Puls-Events**, une plateforme de gestion d'événements culturels en Île-de-France.

Le système :
- Collecte automatiquement les événements via l'API Open Agenda
- Nettoie et filtre les données (Paris uniquement, événements passés < 1 an)
- Segmente les descriptions en chunks
- Encode les textes en vecteurs avec `all-MiniLM-L6-v2`
- Indexe les embeddings dans une base **FAISS** (Flat L2)
- Génère des réponses naturelles via **Mistral-7B**
- Propose une interface web via **Streamlit**

**Périmètre géographique** : Paris & Île-de-France (codes postaux 75xxx)  
**Périmètre temporel** : Événements des 12 derniers mois passés uniquement

---

## 🎯 Objectifs du projet <br>

Ce POC a pour objectifs de : <br>

1. **Démontrer la faisabilité** d'un assistant de recommandation culturelle basé sur RAG
2. **Construire un pipeline complet et reproductible** : collecte → préprocessing → vectorisation → indexation → génération
3. **Valider la pertinence** des réponses générées à partir de données réelles Open Agenda
4. **Poser les bases techniques** d'un déploiement en production (venv, scripts documentés, tests unitaires, README)

---

## 🗂️ Description des fichiers et dossiers <br>

```
rag-events-chatbot/
│
├── .env                        ← Clés API (ne jamais versionner !)
├── .gitignore                  ← Fichiers exclus du versionnement
├── requirements.txt            ← Liste des dépendances Python (pip)
├── README.md                   ← Documentation du projet (ce fichier)
│
├── scripts/
│   ├── fetch_events.py         ← Collecte des événements via API Open Agenda
│   │                              Filtre : Paris (CP 75xxx), < 1 an, passés uniquement
│   ├── build_vector_db.py      ← Chunking, embedding et indexation FAISS
│   │                              Produit : data/index/faiss_index/
│   ├── chatbot.py              ← Chatbot en mode terminal (Mistral API)
│   ├── app.py                  ← Interface web Streamlit (bouton Chercher)
│   └── test_events.py          ← 7 tests unitaires pytest de validation des données
│
└── data/                       ← Dossier généré automatiquement (non versionné)
    ├── events_clean.csv        ← Dataset nettoyé produit par fetch_events.py
    └── index/
        └── faiss_index/        ← Index FAISS produit par build_vector_db.py
            ├── index.faiss     ← Vecteurs binaires
            └── index.pkl       ← Métadonnées et documents
```

---

## ⚙️ Installation et reproduction <br>

### 1. Cloner le projet <br>

```bash
git clone https://github.com/Cheikhafef/rag-events-chatbot.git
cd rag-events-chatbot
```

### 2. Créer l'environnement virtuel <br>

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Installer les dépendances <br>

```bash
pip install -r requirements.txt
```

### 4. Configurer les clés API <br>

Créez un fichier `.env` à la **racine du projet** :

```
MISTRAL_API_KEY=**********************
OPENAGENDA_API_KEY=**********************
```

> 💡 Clé Mistral : [console.mistral.ai](https://console.mistral.ai)  
> 💡 Token HuggingFace (optionnel) : [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Utilisation <br>

Exécutez les scripts dans cet ordre :

### Étape 1 — Collecter les événements <br>

```bash
python scripts/fetch_events.py
```

Résultat attendu :
```
Periode : 2025-03-29  -->  2026-03-29
Total evenements bruts recuperes : 5368
Evenements dans la periode valide (< 1 an, passes) : 360
Dataset sauvegarde : data/events_clean.csv
```

### Étape 2 — Construire la base vectorielle FAISS <br>

```bash
python scripts/build_vector_db.py
```

Résultat attendu :
```
Nombre d'événements chargés : 360
Nombre de chunks créés : 1774
Index FAISS créé — 1774 vecteurs indexés
Index FAISS sauvegardé dans : data/index/faiss_index
```

### Étape 3 — Lancer le chatbot (terminal) <br>

```bash
python scripts/chatbot.py
```

### Étape 4 — Lancer l'interface Streamlit <br>

```bash
streamlit run scripts/app.py
```

Ouvre automatiquement sur [http://localhost:8501](http://localhost:8501)

---

## ✅ Tests unitaires <br>

```bash
python -m pytest scripts/test_events.py -v
```

7 tests sont exécutés et valident les contraintes des étapes 2 et 3 :

| Test | Description |
|------|-------------|
| `test_colonnes_presentes` | Les 7 colonnes attendues sont présentes dans le CSV |
| `test_pas_de_nan` | Aucun NaN dans title, description, start_date |
| `test_dates_periode_valide` | Toutes les dates sont dans la fenêtre < 1 an, passées |
| `test_perimetre_paris` | Tous les événements ont un CP 75xxx ou city = Paris |
| `test_pas_de_doublons` | Aucun doublon sur (title, start_date) |
| `test_index_faiss_non_vide` | L'index FAISS contient au moins 100 vecteurs |
| `test_chunks_faiss_coherents` | >50% des chunks ont une date et localisation valides |

Résultat attendu :
```
7 passed in xx.xxs ✅
```

---

## 🏗️ Architecture RAG <br>

```
PHASE 1 — INDEXATION (offline)
Open Agenda → Préprocessing → Chunking → Embedding  → FAISS Index
    API           Pandas       LangChain   MiniLM-L6    Flat L2

PHASE 2 — QUERY (online)
Question → Retriever FAISS → Contexte → Prompt → Mistral → Réponse
 User        MMR k=15         Chunks    Template   7B-API   Chatbot
```

---

## 🔧 Stack technique <br>

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.11 |
| Collecte | Open Agenda API + Requests |
| Données | Pandas |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (dim=384) |
| Base vectorielle | FAISS Index Flat L2 |
| LLM | Mistral-7B-Instruct-v0.2 via API Mistral AI |
| Orchestration | LangChain + langchain-mistralai |
| Interface | Streamlit |
| Tests | pytest (7 tests unitaires) |
| Environnement | Python venv + python-dotenv |

---

## 📊 Résultats du POC <br>

| Métrique | Valeur |
|----------|--------|
| Événements collectés | 360 |
| Vecteurs FAISS indexés | 1 774 |
| Moy. chunks / événement | 4,9 |
| Temps de recherche | < 1s |
| Tests unitaires | 7/7 passés |

---

## ⚠️ Points d'attention <br>

- Le fichier `.env` ne doit **jamais** être versionné (ajouté dans `.gitignore`)
- Toujours relancer `fetch_events.py` puis `build_vector_db.py` après une mise à jour des données
- Le dossier `data/` est généré automatiquement — ne pas le versionner

---

## 📁 Fichiers générés (non versionnés) <br>

```
data/events_clean.csv
data/index/faiss_index/
.env
venv/
__pycache__/
```

---

## 👤 Auteur

**Ingénieur Data Freelance** — Spécialiste NLP & Bases Vectorielles  
Projet réalisé pour **Puls-Events** — POC RAG 