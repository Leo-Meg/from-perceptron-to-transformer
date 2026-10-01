"""
Réseaux de neurones pour les opérateurs logiques : le TP qui explique *pourquoi*
on empile des couches.

D'où ça vient
-------------
TP « Neural nets for logical operators » (Machine Learning for NLP 1,
Marie Candito, M1). L'énoncé demandait de poser sur papier les poids d'un
classifieur linéaire pour AND et OR, puis de constater qu'aucun choix de poids
ne marche pour XOR, et enfin de construire *à la main* un perceptron multicouche
qui le résout.

Ce TP tient en une demi-page, et pourtant c'est le seul moment de ma formation
où j'ai vu quelqu'un démontrer, avec des nombres qu'on peut vérifier soi-même,
la raison d'être de la profondeur en apprentissage automatique. Je le remets ici
en ouverture du projet parce que la logique est exactement la même que celle qui
justifie l'existence des couches d'un Transformer.

Le cadre
--------
Un opérateur logique binaire est un classifieur : deux entrées booléennes, une
sortie booléenne. On code Faux par -1 et Vrai par +1. Il y a quatre exemples
possibles, et c'est tout le jeu de données :

    (+1, +1), (+1, -1), (-1, +1), (-1, -1)
"""

from __future__ import annotations

import numpy as np

from .algebre import signe, tanh

# Les quatre entrées possibles : c'est l'intégralité de l'univers du problème.
ENTREES = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)

CIBLES = {
    "AND": np.array([1, -1, -1, -1], dtype=float),
    "OR": np.array([1, 1, 1, -1], dtype=float),
    "NAND": np.array([-1, 1, 1, 1], dtype=float),
    "XOR": np.array([-1, 1, 1, -1], dtype=float),
}


class NeuroneLineaire:
    """Un unique neurone : ``signe(w · x + b)``.

    C'est la brique élémentaire. Géométriquement, ``w · x + b = 0`` définit un
    hyperplan (ici une droite du plan), et le neurone répond +1 d'un côté et -1
    de l'autre. Un neurone linéaire ne peut donc apprendre *que* des fonctions
    linéairement séparables.
    """

    def __init__(self, w: np.ndarray, b: float, nom: str = ""):
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)
        self.nom = nom

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Scores avant activation. Forme ``(batch,)``."""
        return X @ self.w + self.b

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Prédiction dans {-1, +1}. Forme ``(batch,)``."""
        return signe(self.scores(X))

    def marge(self, X: np.ndarray, y: np.ndarray) -> float:
        """Marge géométrique minimale : distance du point le plus proche à la droite.

        Question posée en TP : « comment augmenter la marge de votre
        classifieur ? ». La réponse est instructive — multiplier ``w`` et ``b``
        par 10 ne change *rien* à la marge géométrique, puisque la distance d'un
        point à l'hyperplan vaut ``|w·x + b| / ||w||`` : le facteur d'échelle se
        simplifie. Pour vraiment augmenter la marge, il faut changer l'orientation
        de la droite. C'est précisément ce que font les SVM.
        """
        return float(np.min(y * self.scores(X)) / np.linalg.norm(self.w))


# --------------------------------------------------------------------------- #
# Les portes que l'on peut poser à la main
# --------------------------------------------------------------------------- #

# AND : vrai seulement si x1 + x2 = 2, donc x1 + x2 - 1 > 0 uniquement dans ce cas.
PORTE_AND = NeuroneLineaire(w=[1, 1], b=-1, nom="AND")

# OR : faux seulement si x1 + x2 = -2, donc x1 + x2 + 1 < 0 uniquement dans ce cas.
PORTE_OR = NeuroneLineaire(w=[1, 1], b=+1, nom="OR")

# NAND : l'astuce demandée en TP. On ne recalcule rien, on inverse simplement le
# signe des poids ET du biais de AND. Géométriquement c'est la même droite, mais
# les deux demi-plans échangent leurs étiquettes.
PORTE_NAND = NeuroneLineaire(w=[-1, -1], b=+1, nom="NAND")


class ReseauXOR:
    """Perceptron multicouche à une couche cachée de deux neurones, résolvant XOR.

    Pourquoi un neurone seul échoue
    --------------------------------
    XOR vaut +1 pour (+1,-1) et (-1,+1), et -1 pour (+1,+1) et (-1,-1). Les deux
    points positifs sont sur une diagonale, les deux négatifs sur l'autre. Aucune
    droite du plan ne peut séparer une diagonale de l'autre : XOR n'est pas
    linéairement séparable. C'est l'argument de Minsky & Papert (1969) qui a
    gelé la recherche sur les réseaux de neurones pendant une quinzaine d'années.

    La solution
    -----------
    On remarque que ``XOR = AND(OR, NAND)`` : XOR est vrai quand « au moins un »
    est vrai ET « pas les deux » est vrai. On construit donc :

      * un premier neurone caché qui se comporte comme OR ;
      * un second neurone caché qui se comporte comme NAND ;
      * un neurone de sortie qui fait le AND des deux.

    La couche cachée effectue un **changement de représentation** : elle projette
    les quatre points dans un nouvel espace où ils *deviennent* linéairement
    séparables. C'est toute l'idée de l'apprentissage profond, et c'est aussi
    exactement ce que fait chaque bloc d'un Transformer — réécrire les
    représentations pour que la tâche devienne plus facile à la couche suivante.

    Le détail technique de l'activation
    ------------------------------------
    On utilise ``tanh`` avec des poids multipliés par 3 : ``tanh(3) > 0.99`` et
    ``tanh(-3) < -0.99``, donc les sorties de la couche cachée sont numériquement
    presque exactement +1 ou -1. Sans ce facteur d'échelle, ``tanh(1) ≈ 0.76``
    et la couche de sortie, dont les poids sont calibrés pour des entrées ±1,
    se tromperait.
    """

    def __init__(self, gain: float = 3.0):
        self.gain = gain
        # Couche cachée : ligne 0 = OR, ligne 1 = NAND.
        self.W1 = gain * np.array([[1.0, 1.0], [-1.0, -1.0]])  # (2 cachés, 2 entrées)
        self.b1 = gain * np.array([1.0, 1.0])                   # (2 cachés,)
        # Couche de sortie : un AND appliqué aux deux neurones cachés.
        self.W2 = np.array([1.0, 1.0])  # (2 cachés,)
        self.b2 = -1.0

    def couche_cachee(self, X: np.ndarray) -> np.ndarray:
        """Représentation cachée. Forme ``(batch, 2)``."""
        return tanh(X @ self.W1.T + self.b1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Prédiction XOR dans {-1, +1}. Forme ``(batch,)``."""
        h = self.couche_cachee(X)
        return signe(h @ self.W2 + self.b2)


def table_de_verite(nom: str, predictions: np.ndarray) -> str:
    """Affichage lisible d'une table de vérité, avec le verdict."""
    cible = CIBLES[nom]
    lignes = [f"  {nom}"]
    for (x1, x2), pred, gold in zip(ENTREES, predictions, cible):
        marque = "ok" if pred == gold else "ERREUR"
        lignes.append(
            f"    ({x1:+.0f}, {x2:+.0f}) -> prédit {pred:+.0f} | attendu {gold:+.0f}  [{marque}]"
        )
    lignes.append(f"    => {'CORRECT' if np.all(predictions == cible) else 'INCORRECT'}")
    return "\n".join(lignes)


def demonstration_impossibilite_xor() -> None:
    """Preuve par exhaustion (grossière mais convaincante) que XOR est hors de portée.

    Je balaie une grille de couples ``(w1, w2, b)`` et je vérifie qu'aucun ne
    classe correctement les quatre points. Ce n'est évidemment pas une
    démonstration mathématique — c'est une vérification empirique, et elle a le
    mérite d'être immédiatement reproductible.

    La vraie preuve tient en deux lignes : si un neurone classait XOR, alors en
    sommant les inégalités des deux points positifs et des deux points négatifs
    on obtiendrait ``2b > 0`` et ``2b < 0`` simultanément. Contradiction.
    """
    valeurs = np.linspace(-5, 5, 41)
    trouve = False
    for w1 in valeurs:
        for w2 in valeurs:
            for b in valeurs:
                neurone = NeuroneLineaire([w1, w2], b)
                if np.all(neurone(ENTREES) == CIBLES["XOR"]):
                    trouve = True
                    break
    total = len(valeurs) ** 3
    print(f"  {total} triplets (w1, w2, b) testés sur une grille de [-5, 5].")
    print(f"  Solution linéaire trouvée pour XOR : {trouve}")
    print("  Démonstration formelle : en additionnant les contraintes des deux")
    print("  points positifs on obtient 2b > 0 ; celles des deux points négatifs")
    print("  donnent 2b < 0. Aucun b ne satisfait les deux.")


if __name__ == "__main__":
    print("=== 1. Portes réalisables par un seul neurone ===\n")
    for porte in (PORTE_AND, PORTE_OR, PORTE_NAND):
        print(table_de_verite(porte.nom, porte(ENTREES)))
        print(f"    marge géométrique : {porte.marge(ENTREES, CIBLES[porte.nom]):.4f}")
        print()

    print("  Vérification de l'invariance d'échelle de la marge :")
    grand_and = NeuroneLineaire(w=[10, 10], b=-10)
    print(
        f"    w=[1,1], b=-1  -> marge {PORTE_AND.marge(ENTREES, CIBLES['AND']):.4f}\n"
        f"    w=[10,10], b=-10 -> marge {grand_and.marge(ENTREES, CIBLES['AND']):.4f}\n"
        "    Identiques : multiplier les poids ne change pas la géométrie.\n"
    )

    print("=== 2. XOR est hors de portée d'un neurone unique ===\n")
    demonstration_impossibilite_xor()
    print()

    print("=== 3. XOR avec une couche cachée ===\n")
    reseau = ReseauXOR()
    print(table_de_verite("XOR", reseau(ENTREES)))
    print("\n  Représentation apprise par la couche cachée (neurone OR, neurone NAND) :")
    for entree, h in zip(ENTREES, reseau.couche_cachee(ENTREES)):
        print(f"    ({entree[0]:+.0f}, {entree[1]:+.0f}) -> ({h[0]:+.3f}, {h[1]:+.3f})")
    print(
        "\n  Dans ce nouvel espace, les points (+1,+1) et (-1,-1) — ceux dont XOR\n"
        "  doit valoir -1 — sont les seuls à ne pas avoir leurs deux coordonnées\n"
        "  positives. Un simple AND les sépare. La couche cachée n'a pas « appris\n"
        "  XOR » : elle a rendu le problème linéairement séparable."
    )
