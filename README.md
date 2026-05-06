# 🎬 Assistant Films — RAG avec FAISS et Groq

## 📌 Présentation

Assistant Films est un projet de recommandation de films basé sur une architecture RAG (Retrieval-Augmented Generation).

Le système utilise le dataset **TMDB 5000 Movies** pour rechercher des films similaires à une requête utilisateur en langage naturel grâce à des embeddings et à un index vectoriel FAISS.

Les résultats récupérés sont ensuite envoyés à un modèle LLM via l’API Groq afin de générer une réponse naturelle.

---

# ⚙️ Technologies utilisées

- Python
- FAISS
- SentenceTransformers
- Groq API
- Pandas
- NumPy

---

# 📂 Structure du projet

```bash
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
├── rag.py
├── context.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Installation

## 1. Créer un environnement virtuel

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

## 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 3. Ajouter la clé API Groq

Créer un fichier `.env` :

```env
GROQ_API_KEY=votre_cle_api
```

---

# ▶️ Exécution

## Générer l’index vectoriel

```bash
python indexation.py
```

Cette étape :
- charge le dataset ;
- génère les embeddings ;
- crée l’index FAISS.

---

## Lancer l’assistant

```bash
python rag.py
```

---

# 🔍 Fonctionnement

## Indexation

Le script `indexation.py` :
- transforme les films en texte ;
- génère les embeddings ;
- sauvegarde les vecteurs dans FAISS.

## Recherche RAG

Le script `rag.py` :
- convertit la question utilisateur en embedding ;
- recherche les films les plus proches dans FAISS ;
- envoie les résultats au LLM ;
- génère une réponse.

---

# 💬 Exemples de requêtes

```text
Je cherche un thriller psychologique sombre.
```

```text
Un film de science-fiction intelligent.
```

```text
Un film similaire à Interstellar.
```

---

# 🧠 Modèles utilisés

## Embeddings

```text
paraphrase-multilingual-mpnet-base-v2
```

## Recherche vectorielle

```text
FAISS IndexFlatIP
```

## LLM

```text
Llama 3.3 70B via Groq
```

---

# 👨‍💻 Auteurs

Projet réalisé par Ramdane CHABA et Ouissem AOUIMEUR.

---

# 👨‍💻 Ecandré par : 

Hakim HORAIRY
