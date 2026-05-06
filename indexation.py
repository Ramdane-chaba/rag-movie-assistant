import pandas as pd
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# je charge les films depuis le CSV avec pandas
def charger_films(chemin_csv: str) -> pd.DataFrame:
    df = pd.read_csv(chemin_csv)
    print(f"  Nombre de films : {len(df)}")
    print(f"  Colonnes : {df.columns.tolist()}")
    return df


# la colonne genres est en JSON donc je dois la parser
# ex: '[{"id": 18, "name": "Drama"}]' → "Drama"
def extraire_genres(genres_json: str) -> str:
    try:
        genres = json.loads(genres_json)
        return ", ".join([g["name"] for g in genres])
    except Exception:
        return ""


# pareil pour les mots-clés, je prends les 5 premiers
def extraire_keywords(keywords_json: str) -> str:
    try:
        keywords = json.loads(keywords_json)
        return ", ".join([k["name"] for k in keywords[:5]])
    except Exception:
        return ""


# je convertis chaque ligne du CSV en texte pour pouvoir l'embedder
# j'ai choisi title, overview et genres car c'est ce qui décrit le mieux un film
def film_vers_texte(row: pd.Series) -> str:
    titre    = str(row.get("title", "")).strip()
    overview = str(row.get("overview", "")).strip()
    genres   = extraire_genres(str(row.get("genres", "[]")))
    note     = row.get("vote_average", 0)
    votes    = row.get("vote_count", 0)
    date     = str(row.get("release_date", ""))[:4]
    duree    = row.get("runtime", 0)
    langue   = str(row.get("original_language", "")).strip()
    keywords = extraire_keywords(str(row.get("keywords", "[]")))

    # je construis un texte structuré avec toutes les infos importantes
    texte = f"Film : {titre}\n"
    texte += f"Année : {date}\n"
    texte += f"Genres : {genres}\n"
    texte += f"Note : {note}/10 ({int(votes)} votes)\n"
    texte += f"Durée : {int(duree)} minutes\n"
    texte += f"Langue originale : {langue}\n"

    if keywords:
        texte += f"Mots-clés : {keywords}\n"

    if overview and overview != "nan":
        texte += f"Synopsis : {overview}\n"

    return texte.strip()


# je supprime les films sans synopsis et avec peu de votes
# je vérifie qu'il reste au moins 500 films comme demandé dans le sujet
def nettoyer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["overview"].notna() & (df["overview"] != "")]
    df = df[df["vote_count"] >= 10]
    print(f"  Films après nettoyage : {len(df)} (min requis : 500)")
    assert len(df) >= 500, "Pas assez de films ! Vérifiez le fichier CSV."
    return df.reset_index(drop=True)


# je découpe les textes longs en morceaux avec un chevauchement
# pour les films, j'ai mis 1500 caractères pour avoir 1 chunk par film
# priorité de coupure : saut de ligne → espace → coupure forcée
def chunker(texte: str, taille_max: int = 1500, overlap: int = 100) -> list[str]:
    if len(texte) <= taille_max:
        return [texte]

    chunks = []
    debut  = 0

    while debut < len(texte):
        fin = debut + taille_max

        if fin < len(texte):
            coupure = texte.rfind("\n", debut, fin)
            if coupure == -1 or coupure <= debut:
                coupure = texte.rfind(" ", debut, fin)
            if coupure == -1 or coupure <= debut:
                coupure = fin
        else:
            coupure = len(texte)

        chunk = texte[debut:coupure].strip()
        if chunk:
            chunks.append(chunk)

        debut = max(coupure - overlap, debut + 1)

    return chunks


# je transforme chaque chunk en vecteur de nombres
# normalize_embeddings=True est important pour utiliser IndexFlatIP comme similarité cosinus
def embedder_chunks(chunks: list[str], modele) -> np.ndarray:
    vecteurs = modele.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return np.array(vecteurs, dtype=np.float32)


# je crée l'index FAISS avec IndexFlatIP
# avec des vecteurs normalisés, le produit scalaire = similarité cosinus
# un score élevé veut dire que le film est proche de la question
def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    dimension = vecteurs.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)
    print(f"  Index FAISS créé : {index.ntotal} vecteur(s) — dimension {dimension}")
    return index


# je sauvegarde l'index sur disque pour ne pas réindexer à chaque lancement
# l'ordre des chunks doit correspondre à l'ordre des vecteurs dans FAISS
def sauvegarder_index(index: faiss.Index, chunks_avec_meta: list, chemin: str):
    os.makedirs(chemin, exist_ok=True)

    # sauvegarde de l'index FAISS en binaire
    faiss.write_index(index, os.path.join(chemin, "index.faiss"))

    # sauvegarde des textes et métadonnées en JSON
    with open(os.path.join(chemin, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)

    print(f"  Sauvegardé dans '{chemin}/' :")
    print(f"  - index.faiss  ({index.ntotal} vecteurs)")
    print(f"  - chunks.json  ({len(chunks_avec_meta)} chunks)")


if __name__ == "__main__":

    print("=" * 60)
    print("  PHASE 1 — INDEXATION DE LA BASE FILMS")
    print("  Source : tmdb_5000_movies.csv (TMDB / Kaggle)")
    print("=" * 60)

    chemin_csv = "data/tmdb_5000_movies.csv"
    if not os.path.exists(chemin_csv):
        print(f"\n[ERREUR] Fichier introuvable : {chemin_csv}")
        exit(1)

    # si la base existe déjà on ne réindexe pas
    # c'est inutile de recalculer tous les vecteurs si c'est déjà fait
    if os.path.exists("db_films/index.faiss") and os.path.exists("db_films/chunks.json"):
        print("\n✓ Base vectorielle déjà existante — chargement direct.")
        print("  (Supprimez db_films/ pour forcer une réindexation)")
        exit(0)

    # j'utilise le modèle multilingue pour pouvoir poser des questions en français
    # j'ai testé all-mpnet-base-v2 mais les scores étaient moins bons (0.50 vs 0.72)
    print("\n[1/4] Chargement du modèle d'embedding...")
    modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print("  Modèle chargé.")

    # chargement et nettoyage des données
    print("\n[2/4] Chargement et préparation des films...")
    df = charger_films(chemin_csv)
    df = nettoyer_dataframe(df)

    # conversion de chaque film en texte puis en chunks
    # avec taille_max=1500 j'obtiens 1 chunk par film
    print("\n[3/4] Conversion des films en texte et chunking...")
    chunks_avec_meta = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Traitement"):
        texte  = film_vers_texte(row)
        chunks = chunker(texte, taille_max=1500, overlap=100)

        for j, chunk in enumerate(chunks):
            chunks_avec_meta.append({
                "contenu": chunk,
                "metadata": {
                    "titre"  : str(row.get("title", "")),
                    "annee"  : str(row.get("release_date", ""))[:4],
                    "note"   : float(row.get("vote_average", 0)),
                    "votes"  : int(row.get("vote_count", 0)),
                    "genres" : extraire_genres(str(row.get("genres", "[]"))),
                    "langue" : str(row.get("original_language", "")),
                    "film_id": int(row.get("id", idx)),
                    "chunk_id": f"film_{idx}_chunk_{j}",
                    "source" : f"TMDB — film #{row.get('id', idx)}"
                }
            })

    print(f"  Total : {len(chunks_avec_meta)} chunks générés pour {len(df)} films.")

    # création des embeddings et de l'index FAISS
    print("\n[4/4] Création des embeddings et de l'index FAISS...")
    textes   = [c["contenu"] for c in chunks_avec_meta]
    vecteurs = embedder_chunks(textes, modele)

    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks_avec_meta, chemin="db_films")

    print("\n" + "=" * 60)
    print("  INDEXATION TERMINÉE — Lancez maintenant rag.py")
    print("=" * 60)