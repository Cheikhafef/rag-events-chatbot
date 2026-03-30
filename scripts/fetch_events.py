"""
fetch_events.py
===============

Script de collecte et préparation des données d’événements culturels
pour un système RAG (Retrieval-Augmented Generation).

🎯 Objectif :
- Interroger l’API OpenAgenda
- Filtrer les événements situés à Paris (CP 75xxx ou ville = Paris)
- Conserver uniquement les événements passés des 12 derniers mois
- Nettoyer et structurer les données
- Générer un champ textuel "content" optimisé pour le RAG
- Sauvegarder le dataset final en CSV

📥 Entrée :
- API OpenAgenda (via clé API)
- Variables d’environnement (.env)

📤 Sortie :
- data/events_clean.csv

🧠 Étapes principales :
1. Collecte des événements via API
2. Filtrage géographique (Paris uniquement)
3. Sélection des dates passées (< 1 an)
4. Nettoyage des données
5. Création du champ "content" pour le RAG
6. Sauvegarde du dataset

⚙️ Configuration :
- OPENAGENDA_API_KEY doit être défini dans le fichier .env

📌 Remarques :
- Les événements futurs sont exclus volontairement
- Les doublons sont supprimés
- Le champ "content" est utilisé pour l’indexation vectorielle

Projet : RAG Events Chatbot
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# 0  Configuration
API_KEY = os.getenv("OPENAGENDA_API_KEY")

if not API_KEY:
    raise ValueError("ERREUR : clé API OpenAgenda manquante. Vérifiez votre fichier .env")


AGENDA_UIDS = [
    "5790361",   # Info Jeunes Paris
    "96240415",  # Jeunes a Paris
    "78993009",  # Paris&Co
    "56500817",  # OpenAgenda en Ile-de-France (agregateur)
    "86244142",  # Ministere de la Culture
    "24744428",  # Orchestre national d'Ile-de-France
    "49760247",  # Frac Ile-de-France, Le Plateau
    "43070835",  # GL Events Exhibitions - Paris
    "68165804",  # Ile-de-France Nature Animations
    "59610190",  # Evenements sciences en Ile-de-France
]

LIMIT      = 100
MAX_EVENTS = 5000

# Fenetre temporelle : strictement les 12 derniers mois passes
now          = pd.Timestamp.now(tz="UTC")
one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
today        = datetime.now().strftime("%Y-%m-%d")

print("=" * 55)
print("  fetch_events.py — Puls-Events RAG")
print("=" * 55)
print(f"  Periode : {one_year_ago}  -->  {today}")
print(f"  Perimetre : Paris (CP 75xxx ou ville = Paris)")
print("=" * 55 + "\n")

# 1  Collecte via API Open Agenda

all_events = []

print("Recuperation des evenements en cours...")

for uid in AGENDA_UIDS:
    print(f"\n  Agenda {uid}")
    offset = 0

    while True:
        params = {
            "key":          API_KEY,
            "limit":        LIMIT,
            "offset":       offset,
            "timezone":     "Europe/Paris",
            "detailed":     1,
            "startsAfter":  one_year_ago,  # il y a 1 an
            "startsBefore": today,         # jusqu'a aujourd'hui (passes uniquement)
        }

        url = f"https://api.openagenda.com/v2/agendas/{uid}/events"

        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.exceptions.Timeout:
            print("    Timeout — agenda ignore")
            break

        print(f"    Status : {response.status_code}", end="")

        if response.status_code != 200:
            print(f" — Erreur : {response.text[:150]}")
            break

        data   = response.json()
        events = data.get("events", [])
        print(f" — {len(events)} evenements")

        if not events:
            break

        all_events.extend(events)
        offset += LIMIT

        if len(events) < LIMIT or len(all_events) >= MAX_EVENTS:
            break

print(f"\nTotal evenements bruts recuperes : {len(all_events)}")

if not all_events:
    raise ValueError("Aucun evenement recupere — verifiez la cle API.")

# 2  Fonctions utilitaires

def extract_description(event):
    """
    Extrait la meilleure description disponible en français pour un événement.

    Priorité :
    - longDescription
    - description
    - shortDescription
    - summary

    Paramètre :
        event (dict) : événement brut issu de l'API OpenAgenda

    Retour :
        str : description textuelle (peut être vide si non disponible)
    """
    return (
        event.get("longDescription",     {}).get("fr")
        or event.get("description",      {}).get("fr")
        or event.get("shortDescription", {}).get("fr")
        or event.get("summary",          {}).get("fr")
        or ""
    )


def extract_best_past_date(event):
    """
    Retourne la date passée la plus récente d’un événement.

    Règles :
    - Ignore toutes les dates futures
    - Sélectionne la date passée la plus proche du présent
    - Retourne None si aucune date valide

    Paramètre :
        event (dict) : événement contenant une liste de timings

    Retour :
        pandas.Timestamp | None
    """
    timings = event.get("timings", [])
    if not timings:
        return None

    past_dates = []
    for t in timings:
        d = t.get("begin")
        if d:
            try:
                dt = pd.to_datetime(d, utc=True)
                if dt <= now:              #  uniquement les dates passees
                    past_dates.append(dt)
            except Exception:
                pass

    if not past_dates:
        return None

    return max(past_dates)               # la plus recente parmi les passees


def is_paris_event(postal_code: str, city_name: str) -> bool:
    """
    Vérifie si un événement est situé à Paris.

    Critères :
    - Code postal commençant par "75"
    OU
    - Nom de ville contenant "paris"

    Paramètres :
        postal_code (str)
        city_name (str)

    Retour :
        bool
    """
    if postal_code.startswith("75"):
        return True
    if "paris" in city_name.lower():
        return True
    return False


# 3  Construction du dataset

clean_events = []

for event in all_events:
    loc         = event.get("location", {}) or {}
    city_name   = loc.get("city") or ""
    postal_code = str(loc.get("postalCode") or "")

    # Filtre geographique
    if not is_paris_event(postal_code, city_name):
        continue

    # Date passee uniquement
    best_date = extract_best_past_date(event)
    if best_date is None:
        continue

    clean_events.append({
        "title":       event.get("title", {}).get("fr", ""),
        "description": extract_description(event),
        "city":        city_name or "Paris",
        "address":     loc.get("address", ""),
        "postal_code": postal_code,
        "start_date":  best_date,
    })

df = pd.DataFrame(clean_events)
# 4  Nettoyage

df = df.dropna(subset=["title", "description", "start_date"])
df = df[df["title"].str.strip()       != ""]
df = df[df["description"].str.strip() != ""]
df = df.drop_duplicates(subset=["title", "start_date"])
df = df.reset_index(drop=True)

print(f"Apres nettoyage : {len(df)} evenements")

# 5  Double verification de la periode (securite)
df["start_date"] = pd.to_datetime(df["start_date"], utc=True)

limit_past   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
limit_future = pd.Timestamp.now(tz="UTC")

before = len(df)
df = df[df["start_date"] >= limit_past]    # pas plus vieux qu'1 an
df = df[df["start_date"] <= limit_future]  #  strictement dans le passe
after  = len(df)

if before != after:
    print(f"  {before - after} evenement(s) hors periode retire(s)")

print(f"Evenements dans la periode valide (< 1 an, passes) : {len(df)}")


# 6  Champ content pour le RAG

def build_content(row):
   
    """
    Construit le texte descriptif utilisé pour l’indexation RAG.

    Le contenu combine :
    - Titre
    - Lieu
    - Date
    - Description

    Paramètre :
        row (pd.Series) : ligne du DataFrame

    Retour :
        str : texte formaté prêt pour embedding
    
    """
    date_str = (
        row["start_date"].strftime("%d/%m/%Y")
        if pd.notna(row["start_date"])
        else "Date non precisee"
    )
    return (
        f"Evenement : {row['title']}. "
        f"Lieu : {row['address']}, {row['city']}. "
        f"Date : {date_str}. "
        f"Description : {row['description']}"
    )

df["content"] = df.apply(build_content, axis=1)


# 7  Sauvegarde

os.makedirs("data", exist_ok=True)
output_path = "data/events_clean.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\nDataset sauvegarde : {output_path}")
print(f"Nombre final d'evenements : {len(df)}")

# Repartition par mois (sans warning timezone)
df["mois"] = df["start_date"].dt.tz_localize(None).dt.to_period("M")
print("\nRepartition par mois :")
print(df["mois"].value_counts().sort_index().to_string())