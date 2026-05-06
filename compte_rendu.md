# 🎬 Assistant Films — Système RAG de Recommandation Cinématographique

## 📌 Présentation du projet

Ce projet implémente un système de recommandation de films basé sur l’architecture **RAG (Retrieval-Augmented Generation)**. L’objectif principal est de permettre à un utilisateur de poser des questions en langage naturel afin d’obtenir des recommandations de films pertinentes, argumentées et contextualisées.

Le système combine plusieurs composants modernes d’intelligence artificielle :

* un modèle d’**embeddings multilingue** pour représenter les films sous forme de vecteurs numériques ;
* une base vectorielle **FAISS** pour effectuer des recherches sémantiques rapides ;
* un modèle de langage **Llama 3.3 70B** via Groq pour générer des réponses naturelles et détaillées.

Le projet s’appuie sur le dataset **TMDB 5000 Movies** provenant de Kaggle, contenant plusieurs milliers de films accompagnés de leurs métadonnées : synopsis, genres, note moyenne, langue, date de sortie, durée, etc.

---

# 🧠 Objectifs pédagogiques

Ce projet permet de mettre en pratique plusieurs notions importantes liées aux systèmes d’IA générative modernes :

* compréhension du fonctionnement d’un pipeline RAG ;
* création d’une base de données vectorielle ;
* génération d’embeddings textuels ;
* recherche sémantique avec FAISS ;
* interaction avec un LLM ;
* structuration et nettoyage de données ;
* gestion de prompts et contextualisation.

---

# 🏗️ Architecture globale du système

Le fonctionnement du projet se déroule en plusieurs étapes successives.

## 1. Chargement et préparation des données

Le dataset CSV est chargé avec Pandas. Chaque ligne représente un film.

Les colonnes importantes sont ensuite extraites :

* titre ;
* synopsis ;
* genres ;
* date de sortie ;
* note moyenne ;
* langue originale ;
* durée ;
* mots-clés.

Certaines colonnes étant stockées sous forme de JSON dans le CSV, elles sont converties en texte exploitable grâce au module `json`.

Exemple :

```json
[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]
```

devient :

```text
Action, Adventure
```

---

## 2. Transformation en documents textuels

Chaque film est converti en un texte structuré.

Exemple :

```text
Title: Inception
Genres: Science Fiction, Thriller
Overview: A thief who steals corporate secrets through dream-sharing technology...
Vote Average: 8.3
Language: en
Release Date: 2010
```

Cette étape est essentielle car les modèles d’embeddings travaillent sur du texte libre et non directement sur des tableaux CSV.

---

## 3. Génération des embeddings

Le projet utilise le modèle suivant :

```python
paraphrase-multilingual-mpnet-base-v2
```

Ce modèle a été choisi car il comprend plusieurs langues, notamment le français et l’anglais.

Ainsi, un utilisateur peut poser des questions en français même si le dataset est majoritairement en anglais.

Chaque document texte représentant un film est transformé en vecteur numérique dense.

---

## 4. Indexation vectorielle avec FAISS

Les vecteurs sont ensuite stockés dans une base vectorielle FAISS.

Le projet utilise :

```python
faiss.IndexFlatIP
```

Les vecteurs étant normalisés, ce type d’index permet d’obtenir une similarité cosinus exacte.

Deux fichiers sont générés :

```text
index.faiss
chunks.json
```

* `index.faiss` contient les vecteurs ;
* `chunks.json` contient les textes et métadonnées associés.

---

## 5. Recherche sémantique

Quand l’utilisateur pose une question :

```text
Je veux un thriller psychologique avec un retournement inattendu
```

la requête est transformée en embedding.

FAISS recherche ensuite les films les plus proches sémantiquement.

Le système récupère alors les films les plus pertinents.

---

## 6. Génération de réponse avec Groq

Les résultats trouvés sont injectés dans un prompt envoyé au modèle :

```text
Llama 3.3 70B
```

Le LLM génère ensuite une réponse naturelle et argumentée.

Exemple :

```text
Je vous recommande Shutter Island car le film mélange suspense psychologique et narration trompeuse avec une révélation finale marquante.
```

---

# 📂 Structure du projet

```text
PROJET/
│
├── data/
│   └── tmdb_5000_movies.csv
│
├── db_films/
│   ├── index.faiss
│   └── chunks.json
│
├── indexation.py
├── main.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── compte_rendu.md
```

---

# ⚙️ Installation du projet

## 1. Cloner le projet

```bash
git clone <url_du_repo>
cd PROJET
```

---

## 2. Créer un environnement virtuel

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 4. Configurer la clé API Groq

Créer un fichier `.env` :

```env
GROQ_API_KEY=votre_cle_api
```

---

# ▶️ Exécution du projet

## Étape 1 — Génération de la base vectorielle

```bash
python indexation.py
```

Cette étape :

* charge le dataset ;
* prépare les documents ;
* génère les embeddings ;
* construit l’index FAISS ;
* sauvegarde les données.

---

## Étape 2 — Lancement de l’assistant

```bash
python main.py
```

---

# 💬 Exemples de questions

```text
Un film de science-fiction sur l’intelligence artificielle
```

```text
Je veux un film triste mais inspirant
```

```text
Un film similaire à Interstellar
```

```text
Un film d’animation familial récent
```

```text
Un thriller psychologique avec une grosse révélation finale
```

---

# 🔍 Fonctionnalités implémentées

## ✅ Recherche sémantique

Le système ne cherche pas uniquement des mots-clés exacts.

Il comprend également le sens global de la requête.

---

## ✅ Support du français

Le modèle d’embeddings multilingue permet de poser des questions directement en français.

---

## ✅ Recommandations argumentées

Le LLM explique pourquoi chaque film correspond à la demande.

---

## ✅ Évitement des doublons

Le système filtre automatiquement les films répétés dans les résultats.

---

## ✅ Persistance de la base vectorielle

L’index FAISS est sauvegardé localement afin d’éviter une réindexation complète à chaque exécution.

---

# 🧪 Choix techniques

| Élément               | Technologie choisie                   | Justification                        |
| --------------------- | ------------------------------------- | ------------------------------------ |
| Langage               | Python                                | Écosystème IA très riche             |
| Dataset               | TMDB 5000 Movies                      | Base cinématographique complète      |
| Embeddings            | paraphrase-multilingual-mpnet-base-v2 | Compatible français/anglais          |
| Base vectorielle      | FAISS                                 | Recherche rapide et efficace         |
| LLM                   | Llama 3.3 70B via Groq                | Réponses naturelles de haute qualité |
| Similarité            | Cosinus                               | Recherche sémantique robuste         |
| Gestion environnement | python-dotenv                         | Sécurisation des clés API            |

---

# ⚠️ Difficultés rencontrées

## Données JSON dans le CSV

Les colonnes `genres` et `keywords` étaient stockées en JSON imbriqué.

Il a fallu parser correctement ces structures afin d’extraire les informations utiles.

---

## Taille des chunks

Une mauvaise taille de chunk générait trop de fragments inutiles.

La taille a été augmentée afin que chaque film soit représenté par un chunk complet.

---

## Questions multilingues

Le dataset étant majoritairement en anglais, il fallait un modèle capable de comprendre le français sans traduction.

---

## Doublons dans les résultats

FAISS retournait parfois plusieurs chunks appartenant au même film.

Un système de déduplication a été ajouté.

---

# 🚀 Perspectives d’amélioration

Plusieurs améliorations peuvent être envisagées :

* ajout d’une interface web avec Streamlit ou FastAPI ;
* filtrage avancé par année, genre ou note ;
* ajout d’un historique de conversation ;
* utilisation d’une base vectorielle distribuée ;
* mise à jour automatique du dataset ;
* intégration d’affiches et bandes-annonces ;
* fine-tuning spécialisé cinéma.

---

# 📚 Technologies utilisées

* Python
* Pandas
* SentenceTransformers
* FAISS
* NumPy
* Groq API
* dotenv

---

# 📖 Concepts clés du projet

## RAG — Retrieval-Augmented Generation

Un système RAG combine deux éléments :

1. un moteur de recherche documentaire ;
2. un modèle de génération de texte.

Le moteur recherche les informations pertinentes.

Le LLM utilise ensuite ces informations pour produire une réponse contextualisée.

---

## Embeddings

Les embeddings sont des représentations vectorielles de texte.

Deux textes proches sémantiquement auront des vecteurs proches dans l’espace vectoriel.

---

## Base vectorielle

Une base vectorielle stocke les embeddings afin de permettre des recherches rapides par similarité.

---

## Similarité cosinus

La similarité cosinus mesure l’angle entre deux vecteurs.

Plus l’angle est faible, plus les textes sont proches sémantiquement.

---

# 📌 Conclusion

Ce projet démontre la mise en œuvre complète d’un pipeline RAG moderne appliqué à la recommandation de films.

Il combine préparation de données, génération d’embeddings, indexation vectorielle et génération de texte via un LLM.

Le système obtenu est capable de comprendre des requêtes naturelles complexes et de fournir des recommandations cohérentes, contextualisées et argumentées.

Ce projet constitue une excellente introduction pratique aux architectures utilisées aujourd’hui dans les assistants IA modernes et les moteurs de recherche augmentés par génération.
