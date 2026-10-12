"""
Le perceptron multiclasse, implémenté à la main.

D'où ça vient
-------------
TP « Implementing a multiclass Perceptron for document classification »
(Machine Learning for NLP 1, Marie Candito, M1 Linguistique Informatique,
Université Paris Cité). On devait classer des dépêches Reuters représentées
par des sacs de mots.

Pourquoi ce modèle est le bon point de départ
----------------------------------------------
Le perceptron de Rosenblatt (1958) est l'ancêtre direct de la couche finale de
tout classifieur neuronal moderne. Un BERT fine-tuné pour la classification,
c'est *exactement* ça : une matrice de poids et un biais, appliqués à un vecteur
d'entrée. Toute la différence tient dans la façon dont ce vecteur d'entrée est
fabriqué (un sac de mots creux ici, une représentation contextuelle à 768
dimensions là).

Sa règle d'apprentissage est en plus la plus simple qui soit et se démontre en
trois lignes : si on se trompe, on rapproche les poids de la bonne classe de
l'exemple et on éloigne ceux de la classe prédite à tort.

Ce que j'ai ajouté par rapport au rendu de M1
-----------------------------------------------
* la **moyennisation** des poids (averaged perceptron, Freund & Schapire 1999),
  qui améliore franchement la généralisation pour un coût quasi nul ;
* le suivi des courbes d'apprentissage (train/validation) ;
* un arrêt anticipé (early stopping) sur le score de validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HistoriqueEntrainement:
    """Ce que j'enregistre à chaque époque pour tracer les courbes d'apprentissage."""

    exactitude_train: list[float] = field(default_factory=list)
    exactitude_valid: list[float] = field(default_factory=list)
    nb_mises_a_jour: list[int] = field(default_factory=list)


class PerceptronMulticlasse:
    """Perceptron multiclasse « un-contre-tous implicite ».

    Le modèle stocke une matrice ``W`` de forme ``(nb_classes, d)`` : une ligne
    de poids par classe. Pour un exemple ``x``, on calcule un score par classe
    et on prédit l'``argmax``. Il n'y a **pas** de softmax : le perceptron est
    un modèle à marge, pas un modèle probabiliste. C'est précisément ce qui le
    distingue de la régression logistique, qui utilise le même calcul de scores
    mais les interprète comme des log-probabilités.

    Args:
        nb_classes: nombre de classes de sortie.
        moyenne: si vrai, on renvoie les poids *moyennés* sur toutes les mises à
            jour au lieu des derniers poids. C'est presque toujours meilleur :
            les derniers poids dépendent trop de l'ordre des derniers exemples.
        graine: graine aléatoire, pour que mes résultats soient reproductibles.
    """

    def __init__(self, nb_classes: int, moyenne: bool = True, graine: int = 42):
        self.nb_classes = nb_classes
        self.moyenne = moyenne
        self.rng = np.random.default_rng(graine)

        self.W: np.ndarray | None = None  # (nb_classes, d)
        self.b: np.ndarray | None = None  # (nb_classes,)

    # ------------------------------------------------------------------ #
    # Prédiction
    # ------------------------------------------------------------------ #

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Scores bruts pour un lot d'exemples. Forme de sortie ``(batch, nb_classes)``.

        Note: j'écris ``X @ self.W.T`` et non ``self.W @ X`` — c'est la
        convention « batch en premier ». Chaque ligne de la sortie correspond à
        un exemple, chaque colonne à une classe.
        """
        assert self.W is not None, "Le modèle doit être entraîné avant de prédire."
        return X @ self.W.T + self.b

    def predire(self, X: np.ndarray) -> np.ndarray:
        """Classe prédite pour chaque exemple. Forme ``(batch,)``."""
        return np.argmax(self.scores(X), axis=1)

    def evaluer(self, X: np.ndarray, y: np.ndarray) -> float:
        """Exactitude (accuracy) sur un jeu de données, entre 0 et 1."""
        return float(np.mean(self.predire(X) == y))

    # ------------------------------------------------------------------ #
    # Apprentissage
    # ------------------------------------------------------------------ #

    def entrainer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        nb_epoques: int = 20,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
        patience: int | None = None,
        verbeux: bool = True,
    ) -> HistoriqueEntrainement:
        """Algorithme d'apprentissage du perceptron.

        Pour chaque exemple, dans un ordre aléatoire :
          1. on prédit ;
          2. si la prédiction est correcte, on ne touche à rien ;
          3. sinon, on **ajoute** l'exemple aux poids de la classe correcte et
             on le **retranche** des poids de la classe prédite à tort.

        C'est tout. Il n'y a ni taux d'apprentissage ni fonction de coût
        explicite. On peut montrer que si les données sont linéairement
        séparables, l'algorithme converge en un nombre fini d'étapes
        (théorème de convergence de Novikoff, 1962).

        Args:
            patience: si renseigné, on arrête après ``patience`` époques sans
                amélioration du score de validation.
        """
        n, d = X_train.shape
        self.W = np.zeros((self.nb_classes, d))
        self.b = np.zeros(self.nb_classes)

        # Accumulateurs pour la version moyennée.
        W_cumul = np.zeros_like(self.W)
        b_cumul = np.zeros_like(self.b)
        nb_accumulations = 0

        historique = HistoriqueEntrainement()
        meilleur_score = -1.0
        epoques_sans_progres = 0

        for epoque in range(nb_epoques):
            ordre = self.rng.permutation(n)
            mises_a_jour = 0

            for i in ordre:
                x_i, y_i = X_train[i], y_train[i]
                scores = self.W @ x_i + self.b
                prediction = int(np.argmax(scores))

                if prediction != y_i:
                    # On « pousse » l'hyperplan de la bonne classe vers l'exemple
                    # et celui de la mauvaise classe dans la direction opposée.
                    self.W[y_i] += x_i
                    self.b[y_i] += 1.0
                    self.W[prediction] -= x_i
                    self.b[prediction] -= 1.0
                    mises_a_jour += 1

                # On accumule à *chaque* exemple (et non à chaque mise à jour) :
                # c'est la définition exacte du perceptron moyenné.
                W_cumul += self.W
                b_cumul += self.b
                nb_accumulations += 1

            # Poids effectivement utilisés pour l'évaluation de cette époque.
            W_courant, b_courant = self.W, self.b
            if self.moyenne:
                self.W = W_cumul / nb_accumulations
                self.b = b_cumul / nb_accumulations

            acc_train = self.evaluer(X_train, y_train)
            historique.exactitude_train.append(acc_train)
            historique.nb_mises_a_jour.append(mises_a_jour)

            acc_valid = float("nan")
            if X_valid is not None and y_valid is not None:
                acc_valid = self.evaluer(X_valid, y_valid)
                historique.exactitude_valid.append(acc_valid)

            if verbeux:
                msg = (
                    f"époque {epoque + 1:>3} | mises à jour : {mises_a_jour:>6} "
                    f"| train {acc_train:.4f}"
                )
                if not np.isnan(acc_valid):
                    msg += f" | valid {acc_valid:.4f}"
                print(msg)

            # On restaure les poids « bruts » pour continuer l'apprentissage :
            # la moyenne n'est qu'une vue en lecture, elle ne doit pas polluer
            # l'état interne de l'algorithme.
            self.W, self.b = W_courant, b_courant

            if patience is not None and not np.isnan(acc_valid):
                if acc_valid > meilleur_score:
                    meilleur_score = acc_valid
                    epoques_sans_progres = 0
                else:
                    epoques_sans_progres += 1
                    if epoques_sans_progres >= patience:
                        if verbeux:
                            print(f"Arrêt anticipé à l'époque {epoque + 1}.")
                        break

        if self.moyenne:
            self.W = W_cumul / nb_accumulations
            self.b = b_cumul / nb_accumulations

        return historique


def matrice_sac_de_mots(
    documents: list[list[str]], vocabulaire: dict[str, int] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Vectorisation « sac de mots » (comptes bruts), écrite à la main.

    C'est l'équivalent de ``sklearn.feature_extraction.text.CountVectorizer``,
    que j'ai utilisé en TP. Je le réécris ici parce que c'est utile de voir que
    derrière ce nom se cache uniquement : un dictionnaire mot -> indice, puis un
    comptage.

    Ce que cette représentation perd — et c'est énorme — c'est **l'ordre des
    mots**. « le chien mord l'homme » et « l'homme mord le chien » ont
    exactement le même vecteur. C'est la limite qui motivera tout le reste du
    projet : les RNN, puis l'attention, existent pour réintroduire l'ordre.

    Args:
        documents: liste de documents, chacun déjà segmenté en tokens.
        vocabulaire: si fourni, on le réutilise (indispensable pour vectoriser
            le test avec le vocabulaire du train). Les mots inconnus sont
            simplement ignorés.

    Returns:
        ``(X, vocabulaire)`` avec ``X`` de forme ``(nb_documents, taille_vocab)``.
    """
    if vocabulaire is None:
        mots = sorted({mot for doc in documents for mot in doc})
        vocabulaire = {mot: i for i, mot in enumerate(mots)}

    X = np.zeros((len(documents), len(vocabulaire)), dtype=np.float64)
    for i, doc in enumerate(documents):
        for mot in doc:
            j = vocabulaire.get(mot)
            if j is not None:  # mot hors-vocabulaire : ignoré
                X[i, j] += 1.0
    return X, vocabulaire


if __name__ == "__main__":
    print("=== Perceptron multiclasse sur données synthétiques ===\n")

    rng = np.random.default_rng(0)
    nb_classes, d, n = 4, 30, 900

    # Je fabrique des données presque séparables : un « prototype » par classe,
    # plus un bruit gaussien assez fort pour que la tâche ne soit pas triviale
    # (sinon les deux variantes du perceptron atteignent 100 % et on ne voit
    # plus l'intérêt de la moyennisation).
    prototypes = rng.normal(size=(nb_classes, d))
    y = rng.integers(0, nb_classes, size=n)
    X = prototypes[y] + rng.normal(size=(n, d)) * 2.5

    X_train, y_train = X[:700], y[:700]
    X_valid, y_valid = X[700:], y[700:]

    for moyenne in (False, True):
        etiquette = "moyenné" if moyenne else "classique"
        print(f"--- Perceptron {etiquette} ---")
        modele = PerceptronMulticlasse(nb_classes, moyenne=moyenne)
        modele.entrainer(X_train, y_train, nb_epoques=5, X_valid=X_valid,
                         y_valid=y_valid, verbeux=False)
        print(f"exactitude validation : {modele.evaluer(X_valid, y_valid):.4f}\n")

    print("=== Sac de mots ===\n")
    corpus = [
        "le chien mord l' homme".split(),
        "l' homme mord le chien".split(),
        "le chat dort".split(),
    ]
    X_bow, vocab = matrice_sac_de_mots(corpus)
    print(f"vocabulaire : {vocab}")
    print(f"matrice :\n{X_bow.astype(int)}")
    print(
        "\nLes deux premières lignes sont identiques : le sac de mots ne peut "
        "pas distinguer\n« le chien mord l'homme » de « l'homme mord le chien ». "
        "C'est le problème\nque l'attention résoudra à la fin de ce projet."
    )
