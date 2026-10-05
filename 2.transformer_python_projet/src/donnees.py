"""
Jeux de données jouets, embarqués dans le dépôt.

Pourquoi je n'utilise aucun téléchargement
-------------------------------------------
Tous mes TP commençaient par un `wget` vers le serveur de l'université ou vers
Google Drive. Deux ans plus tard, la moitié de ces URL sont mortes et les
notebooks ne s'exécutent plus. J'embarque donc ici des corpus minuscules mais
réels, directement dans le code : le dépôt reste exécutable tel quel, pour
toujours et sans réseau.

La tâche principale — l'identification de la langue — est celle du TP
« A first classifier in PyTorch: language identification using a bag of word
input » (Machine Learning for NLP 1, M1). C'est un excellent banc d'essai :
la tâche est facile, donc si le modèle échoue, c'est que le code est faux.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Corpus multilingue minimal
# --------------------------------------------------------------------------- #

PHRASES: dict[str, list[str]] = {
    "français": [
        "je pense donc je suis",
        "la linguistique informatique est un domaine passionnant",
        "le chat dort sur le tapis du salon",
        "nous avons étudié les modèles de langue pendant deux ans",
        "il fait beau ce matin dans le quartier",
        "la traduction automatique a beaucoup progressé récemment",
        "les étudiants du master travaillent sur des corpus annotés",
        "elle a écrit un mémoire sur la sémantique formelle",
        "ce modèle apprend une représentation des mots",
        "on ne peut pas comparer ces deux corpus directement",
        "le sens d'un mot dépend de son contexte",
        "les réseaux de neurones ont changé notre discipline",
        "j'ai passé beaucoup de temps sur ce devoir",
        "la grammaire de cette langue est très irrégulière",
        "nous devons annoter mille phrases avant vendredi",
        "le professeur explique la structure de la phrase",
        "chaque mot reçoit une étiquette morphosyntaxique",
        "il faut évaluer le système sur un corpus de test",
        "cette méthode donne de meilleurs résultats que la précédente",
        "les données sont réparties en trois ensembles distincts",
        "le programme calcule la probabilité de chaque séquence",
        "on observe une amélioration nette de la performance",
    ],
    "anglais": [
        "i think therefore i am",
        "computational linguistics is a fascinating field",
        "the cat sleeps on the living room carpet",
        "we have studied language models for two years",
        "the weather is nice this morning in the neighbourhood",
        "machine translation has improved a lot recently",
        "the master students work on annotated corpora",
        "she wrote a thesis about formal semantics",
        "this model learns a representation of words",
        "we cannot compare these two corpora directly",
        "the meaning of a word depends on its context",
        "neural networks have changed our discipline",
        "i spent a lot of time on this assignment",
        "the grammar of this language is very irregular",
        "we must annotate a thousand sentences before friday",
        "the professor explains the structure of the sentence",
        "each word receives a morphosyntactic label",
        "we need to evaluate the system on a test corpus",
        "this method gives better results than the previous one",
        "the data is split into three distinct sets",
        "the program computes the probability of each sequence",
        "we observe a clear improvement in performance",
    ],
    "espagnol": [
        "pienso luego existo",
        "la lingüística computacional es un campo apasionante",
        "el gato duerme sobre la alfombra del salón",
        "hemos estudiado modelos de lengua durante dos años",
        "hace buen tiempo esta mañana en el barrio",
        "la traducción automática ha mejorado mucho recientemente",
        "los estudiantes del máster trabajan con corpus anotados",
        "ella escribió una tesis sobre semántica formal",
        "este modelo aprende una representación de las palabras",
        "no podemos comparar estos dos corpus directamente",
        "el sentido de una palabra depende de su contexto",
        "las redes neuronales han cambiado nuestra disciplina",
        "he pasado mucho tiempo en este trabajo",
        "la gramática de esta lengua es muy irregular",
        "debemos anotar mil frases antes del viernes",
        "el profesor explica la estructura de la frase",
        "cada palabra recibe una etiqueta morfosintáctica",
        "hay que evaluar el sistema en un corpus de prueba",
        "este método da mejores resultados que el anterior",
        "los datos se reparten en tres conjuntos distintos",
        "el programa calcula la probabilidad de cada secuencia",
        "observamos una clara mejora del rendimiento",
    ],
}


def ngrammes_caracteres(texte: str, n: int = 3) -> list[str]:
    """Découpe un texte en n-grammes de caractères, avec bornes de mot.

    Pourquoi les n-grammes de caractères plutôt que les mots pour
    l'identification de la langue : ils captent la **phonotactique** et
    l'orthographe (les suites `_th`, `ing`, `eur_`, `ción`), qui sont bien plus
    discriminantes que le lexique et bien plus robustes aux mots inconnus. Un
    identifieur de langue par n-grammes de caractères reste, aujourd'hui encore,
    une base extrêmement solide.
    """
    texte = "_" + texte.replace(" ", "_") + "_"
    return [texte[i:i + n] for i in range(len(texte) - n + 1)]


def corpus_identification_langue(
    n: int = 3, graine: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[str]]:
    """Construit un jeu train/test pour l'identification de la langue.

    Représentation : sac de n-grammes de caractères, normalisé en fréquences
    relatives (sinon les phrases longues auraient des vecteurs de norme plus
    grande, ce qui biaiserait le modèle vers elles).

    Returns:
        ``(X_train, y_train, X_test, y_test, vocabulaire, noms_de_classes)``
    """
    rng = np.random.default_rng(graine)
    langues = sorted(PHRASES)

    # Découpage **stratifié** : je répartis chaque langue séparément entre train
    # et test. Sur un corpus aussi petit, un découpage aléatoire global peut
    # facilement produire un test sans aucun exemple d'une des langues — ce qui
    # rend le score ininterprétable. C'est exactement le rôle du paramètre
    # `stratify` de `sklearn.model_selection.train_test_split`.
    train: list[tuple[list[str], int]] = []
    test: list[tuple[list[str], int]] = []
    for i, langue in enumerate(langues):
        phrases = list(PHRASES[langue])
        rng.shuffle(phrases)
        coupe = int(0.75 * len(phrases))
        train += [(ngrammes_caracteres(p, n), i) for p in phrases[:coupe]]
        test += [(ngrammes_caracteres(p, n), i) for p in phrases[coupe:]]
    rng.shuffle(train)
    rng.shuffle(test)

    # Le vocabulaire est construit **uniquement sur le train**. C'est un point
    # méthodologique essentiel : le construire sur tout le corpus revient à
    # laisser fuiter de l'information du test vers le train.
    vocabulaire: dict[str, int] = {}
    for grammes, _ in train:
        for g in grammes:
            vocabulaire.setdefault(g, len(vocabulaire))

    def vectoriser(donnees) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((len(donnees), len(vocabulaire)))
        y = np.zeros(len(donnees), dtype=int)
        for i, (grammes, etiquette) in enumerate(donnees):
            for g in grammes:
                j = vocabulaire.get(g)
                if j is not None:  # n-gramme inconnu : ignoré
                    X[i, j] += 1.0
            total = X[i].sum()
            if total > 0:
                X[i] /= total  # fréquences relatives
            y[i] = etiquette
        return X, y

    X_train, y_train = vectoriser(train)
    X_test, y_test = vectoriser(test)
    return X_train, y_train, X_test, y_test, vocabulaire, langues


if __name__ == "__main__":
    from .reseau import Lineaire, ReLU, Sequentiel, entrainer

    print("=== Identification de la langue par sac de n-grammes de caractères ===\n")

    X_train, y_train, X_test, y_test, vocab, langues = corpus_identification_langue(n=3)
    print(f"langues        : {langues}")
    print(f"train          : {X_train.shape}")
    print(f"test           : {X_test.shape}")
    print(f"vocabulaire    : {len(vocab)} trigrammes de caractères\n")

    rng = np.random.default_rng(0)
    modele = Sequentiel(
        Lineaire(X_train.shape[1], 32, rng=rng), ReLU(),
        Lineaire(32, len(langues), rng=rng),
    )
    # J'entraîne sans jamais montrer le test au modèle — pas même pour l'arrêt
    # anticipé, qui serait déjà une forme de sélection sur le test.
    entrainer(modele, X_train, y_train,
              nb_epoques=200, taille_lot=8, lr=0.01, patience=None, verbeux=False)

    modele.eval()
    predictions = np.argmax(modele.avant(X_test), axis=1)
    print(f"exactitude sur le test : {np.mean(predictions == y_test):.4f}\n")

    print("Prédictions sur des phrases jamais vues :")
    for phrase in [
        "le corpus contient des phrases annotées",
        "the corpus contains annotated sentences",
        "el corpus contiene frases anotadas",
    ]:
        x = np.zeros((1, len(vocab)))
        for g in ngrammes_caracteres(phrase, 3):
            j = vocab.get(g)
            if j is not None:
                x[0, j] += 1.0
        if x.sum() > 0:
            x /= x.sum()
        print(f"  « {phrase} » -> {langues[int(np.argmax(modele.avant(x)))]}")
