import json
import os
import re
import time

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------------------
# Chargement des variables d'environnement
# -------------------------------------------------------------------
load_dotenv()


# -------------------------------------------------------------------
# Lecture du fichier de contexte
# -------------------------------------------------------------------
def read_file(file_path: str) -> str:
    """
    Lit le contenu d'un fichier texte.

    Args:
        file_path: chemin vers le fichier

    Returns:
        Contenu du fichier
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------------------------------------------
# Chargement de la base FAISS
# -------------------------------------------------------------------
def charger_index(chemin: str):
    """
    Recharge l'index FAISS et les chunks depuis le disque.

    Args:
        chemin: dossier contenant index.faiss et chunks.json

    Returns:
        Tuple (index FAISS, liste des chunks)
    """
    chemin_index = os.path.join(chemin, "index.faiss")
    chemin_chunks = os.path.join(chemin, "chunks.json")

    if not os.path.exists(chemin_index) or not os.path.exists(chemin_chunks):
        raise FileNotFoundError(
            f"Base introuvable dans '{chemin}/'. "
            "Lancez d'abord indexation.py."
        )

    index = faiss.read_index(chemin_index)

    with open(chemin_chunks, "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)

    print(
        f"  Base chargée : "
        f"{index.ntotal} vecteurs — "
        f"{len(chunks_avec_meta)} chunks"
    )

    return index, chunks_avec_meta


# -------------------------------------------------------------------
# Détection filtre langue
# -------------------------------------------------------------------
def detecter_filtre_langue(question: str) -> str | None:
    """
    Détecte si l'utilisateur veut une langue spécifique.

    Returns:
        'fr', 'en' ou None
    """
    question_lower = question.lower()

    if any(
        mot in question_lower
        for mot in [
            "français",
            "francais",
            "française",
            "en français",
            "vo française"
        ]
    ):
        return "fr"

    if any(
        mot in question_lower
        for mot in [
            "anglais",
            "english",
            "américain",
            "americain"
        ]
    ):
        return "en"

    return None


# -------------------------------------------------------------------
# Filtrage métier par année
# -------------------------------------------------------------------
def filtrer_par_annee(question: str, resultats: list) -> list:
    """
    Filtre les résultats selon une contrainte temporelle :
    - avant XXXX
    - après XXXX
    - de XXXX
    - année XXXX
    - sorti en XXXX
    """

    question_lower = question.lower()

    # avant XXXX
    match_avant = re.search(r"avant\s+(\d{4})", question_lower)

    if match_avant:
        annee_max = int(match_avant.group(1))

        resultats = [
            r for r in resultats
            if r["metadata"].get("annee")
            and int(r["metadata"]["annee"]) < annee_max
        ]

    # après XXXX
    match_apres = re.search(r"après\s+(\d{4})", question_lower)

    if match_apres:
        annee_min = int(match_apres.group(1))

        resultats = [
            r for r in resultats
            if r["metadata"].get("annee")
            and int(r["metadata"]["annee"]) > annee_min
        ]

    # année exacte
    # exemples :
    # - film de 2010
    # - film sorti en 2004
    # - film de l'année 1999
    match_exact = re.search(
        r"(de|année|annee|sorti en|sortie en)\s+(\d{4})",
        question_lower
    )

    if match_exact:

        annee_exacte = int(match_exact.group(2))

        resultats = [
            r for r in resultats
            if r["metadata"].get("annee")
            and int(r["metadata"]["annee"]) == annee_exacte
        ]

    return resultats

# -------------------------------------------------------------------
# Recherche vectorielle
# -------------------------------------------------------------------
def retrieve(
    question: str,
    modele,
    index,
    chunks_avec_meta: list,
    n: int = 5
) -> list[dict]:
    """
    Recherche les films les plus pertinents.

    Returns:
        Liste des résultats pertinents
    """

    vecteur_question = modele.encode(
        [question],
        normalize_embeddings=True
    ).astype(np.float32)

    # Recherche FAISS
    scores, indices = index.search(vecteur_question, n * 3)

    filtre_langue = detecter_filtre_langue(question)

    resultats = []
    titres_vus = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks_avec_meta[idx]

        titre = chunk["metadata"].get("titre", "")

        # Éviter les doublons
        if titre in titres_vus:
            continue

        # Filtre langue
        if filtre_langue:
            if chunk["metadata"].get("langue", "") != filtre_langue:
                continue

        resultats.append({
            "contenu": chunk["contenu"],
            "metadata": chunk["metadata"],
            "score": float(score)
        })

        titres_vus.add(titre)

        if len(resultats) >= n:
            break

    # Filtrage métier par année
    resultats = filtrer_par_annee(question, resultats)

    return resultats


# -------------------------------------------------------------------
# Construction du contexte
# -------------------------------------------------------------------
def build_context(
    question: str,
    modele,
    index,
    chunks_avec_meta: list
):
    """
    Construit le contexte envoyé au LLM.
    """

    context = read_file("context.txt")

    chuncks = retrieve(
        question,
        modele,
        index,
        chunks_avec_meta,
        n=5
    )

    chuncks_texte = []

    for i, chunk in enumerate(chuncks, 1):

        meta = chunk["metadata"]

        chuncks_texte.append(
            f"Film {i} : "
            f"{meta.get('titre')} "
            f"({meta.get('annee')}) "
            f"— Note : {meta.get('note')}/10 "
            f"— Genres : {meta.get('genres')} "
            f"— ID : {meta.get('film_id')}\n"
            f"{chunk['contenu']}"
        )

    full_context = context.replace(
        "{{Chuncks}}",
        "\n\n".join(chuncks_texte)
    )

    return full_context, chuncks


# -------------------------------------------------------------------
# Génération de réponse
# -------------------------------------------------------------------
def answer_question(
    question: str,
    modele,
    index,
    chunks_avec_meta: list
):
    """
    Génère une réponse avec Groq.
    """

    full_context, chuncks = build_context(
        question,
        modele,
        index,
        chunks_avec_meta
    )

    # Refus intelligent
    if not chuncks or chuncks[0]["score"] < 0.45:
        return (
            "Je ne trouve pas cette information dans ma base de connaissances.",
            chuncks
        )

    client = Groq(
        api_key=os.environ["GROQ_API_KEY"]
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": full_context,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024
    )

    return chat_completion.choices[0].message.content, chuncks


# -------------------------------------------------------------------
# Interface CLI
# -------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("🎬 ASSISTANT FILMS — RAG avec FAISS + Groq")
    print("📚 Base : TMDB 5000 Movies Dataset")
    print("=" * 70)

    # Chargement modèle
    print("\n⏳ Chargement du modèle d'embedding...")

    modele = SentenceTransformer(
        "paraphrase-multilingual-mpnet-base-v2"
    )

    # Chargement base
    print("⏳ Chargement de la base vectorielle...")

    index, chunks_avec_meta = charger_index("db_films")

    print("\n✅ Système prêt.")
    print("Tapez 'quit' pour quitter.\n")

    print("💡 Exemples de questions :")
    print("  • Un thriller psychologique avec un retournement inattendu ?")
    print("  • Un film d'animation familial sorti après 2010 ?")
    print("  • Un film comme Inception mais plus accessible ?")
    print("  • Un film de science-fiction sur l'IA en français ?")
    print("  • Un film d'amour avant 2009 ?")

    print("\n" + "=" * 70)

    while True:

        try:
            question = input("\n🎥 Votre question : ").strip()

            # Quitter
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Au revoir !")
                break

            # Question vide
            if not question:
                print("⚠️ Veuillez saisir une question.")
                continue

            debut = time.time()

            # Génération réponse
            reponse, chuncks = answer_question(
                question,
                modele,
                index,
                chunks_avec_meta
            )

            temps_execution = time.time() - debut

            print("\n" + "-" * 70)
            print("🤖 Réponse :\n")

            print(reponse)

            # Affichage des sources
            if chuncks:

                print("\n📚 Films utilisés :")

                for chunk in chuncks:

                    meta = chunk["metadata"]

                    print(
                        f"  • {meta.get('titre')} "
                        f"({meta.get('annee')}) "
                        f"— Note : {meta.get('note')}/10 "
                        f"— Score : {chunk['score']:.3f}"
                    )

            print(f"\n⏱️ Temps de réponse : {temps_execution:.2f} sec")

            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme.")
            break

        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            print("-" * 70)