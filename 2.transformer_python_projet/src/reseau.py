"""
Un mini-framework de réseaux de neurones : couches, rétropropagation, optimiseurs.

L'idée
------
Dans les TP de M1 et M2, on écrivait `loss.backward()` et PyTorch faisait le
reste. C'est très pratique, et c'est aussi une boîte noire. Je réécris donc ici
le mécanisme complet — passe avant, passe arrière, mise à jour des paramètres —
en NumPy pur, sans différentiation automatique.

L'architecture que je choisis est celle des frameworks « à la Caffe » : chaque
couche est un objet qui sait faire deux choses.

  * `avant(x)` : calcule la sortie et **mémorise** ce dont la passe arrière aura
    besoin.
  * `arriere(grad_sortie)` : reçoit le gradient du coût par rapport à sa sortie,
    calcule le gradient par rapport à ses paramètres (stocké dans `self.grads`)
    et renvoie le gradient par rapport à son entrée.

La rétropropagation n'est alors rien d'autre que la règle de dérivation des
fonctions composées appliquée de la dernière couche vers la première.

Ce que ça m'a appris
---------------------
Deux choses que je n'avais pas vraiment comprises avant de l'écrire :

1. `zero_grad()` n'est pas une formalité. Les gradients s'**accumulent** par
   défaut ; si on oublie de les remettre à zéro, on optimise une somme de
   gradients de lots différents. J'ai reproduit le bug volontairement dans les
   tests pour voir l'effet.
2. La passe arrière d'une couche linéaire fait apparaître naturellement la
   **transposée** de la matrice de poids. Ce n'est pas une astuce, c'est la
   conséquence directe du fait que la dérivée d'une application linéaire est
   son adjoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .algebre import log_softmax, relu, relu_derivee, softmax, tanh, tanh_derivee


# --------------------------------------------------------------------------- #
# Interface commune
# --------------------------------------------------------------------------- #


class Couche(ABC):
    """Interface d'une couche : une passe avant, une passe arrière, des paramètres."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.cache: dict[str, np.ndarray] = {}
        self.entrainement = True

    @abstractmethod
    def avant(self, x: np.ndarray) -> np.ndarray:
        """Calcule la sortie et met en cache ce dont `arriere` aura besoin."""

    @abstractmethod
    def arriere(self, grad_sortie: np.ndarray) -> np.ndarray:
        """Renvoie le gradient par rapport à l'entrée ; remplit `self.grads`."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.avant(x)

    def zero_grad(self) -> None:
        for nom in self.grads:
            self.grads[nom].fill(0.0)


# --------------------------------------------------------------------------- #
# Couche dense
# --------------------------------------------------------------------------- #


class Lineaire(Couche):
    """Couche entièrement connectée : ``y = x @ W + b``.

    C'est l'équivalent de `torch.nn.Linear`, que le TP 1 de M2 nous demandait
    justement de réimplémenter (« Obviously, you are not allowed to use
    torch.nn.Linear »).

    Initialisation
    --------------
    L'énoncé du TP mettait en garde : « ne pas initialiser tous les paramètres de
    façon à ce qu'ils soient tous positifs ». J'utilise ici l'initialisation de
    **He** (2015), adaptée à ReLU : écart-type ``sqrt(2 / d_entree)``.

    Pourquoi ce facteur : si les poids sont trop grands, la variance des
    activations explose couche après couche ; trop petits, elle s'effondre et le
    gradient disparaît. He montre qu'avec ReLU (qui annule la moitié des
    activations) ce facteur maintient la variance à peu près constante en
    profondeur. Pour tanh ou sigmoïde on prendrait Xavier/Glorot,
    ``sqrt(1 / d_entree)``.
    """

    def __init__(self, d_entree: int, d_sortie: int, biais: bool = True,
                 rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or np.random.default_rng()
        ecart_type = np.sqrt(2.0 / d_entree)

        self.params["W"] = rng.normal(0.0, ecart_type, size=(d_entree, d_sortie))
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.avec_biais = biais
        if biais:
            self.params["b"] = np.zeros(d_sortie)
            self.grads["b"] = np.zeros(d_sortie)

    def avant(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x  # nécessaire pour dL/dW
        y = x @ self.params["W"]
        if self.avec_biais:
            y = y + self.params["b"]
        return y

    def arriere(self, grad_sortie: np.ndarray) -> np.ndarray:
        """Trois gradients à calculer, un par « entrée » de l'opération.

        Avec ``y = x @ W + b`` :

            dL/dW = x^T @ dL/dy     — forme (d_entree, d_sortie)
            dL/db = somme sur le batch de dL/dy
            dL/dx = dL/dy @ W^T     — forme (batch, d_entree)

        Le ``+=`` (et non ``=``) est volontaire : c'est ce qui rend l'accumulation
        de gradients possible, et c'est aussi ce qui rend `zero_grad()`
        obligatoire.
        """
        x = self.cache["x"]
        self.grads["W"] += x.T @ grad_sortie
        if self.avec_biais:
            self.grads["b"] += grad_sortie.sum(axis=0)
        return grad_sortie @ self.params["W"].T


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #


class ReLU(Couche):
    """ReLU. Aucun paramètre : la passe arrière se contente de masquer le gradient."""

    def avant(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        return relu(x)

    def arriere(self, grad_sortie: np.ndarray) -> np.ndarray:
        # Le gradient ne passe que là où l'entrée était positive : c'est
        # littéralement une porte ouverte/fermée, d'où le nom « rectified ».
        return grad_sortie * relu_derivee(self.cache["x"])


class Tanh(Couche):
    """Tangente hyperbolique."""

    def avant(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x
        return tanh(x)

    def arriere(self, grad_sortie: np.ndarray) -> np.ndarray:
        return grad_sortie * tanh_derivee(self.cache["x"])


class Dropout(Couche):
    """Dropout (Srivastava et al., 2014) : régularisation par extinction aléatoire.

    Pendant l'entraînement, on met à zéro chaque activation avec probabilité
    ``p``, puis on divise les survivantes par ``1 - p`` (« inverted dropout »).
    Cette division permet de ne **rien** faire à l'inférence : l'espérance des
    activations est la même dans les deux régimes.

    Intuition : on empêche le réseau de dépendre d'un neurone particulier, donc
    on l'oblige à répartir l'information. C'est une forme d'apprentissage
    d'ensemble à coût nul.
    """

    def __init__(self, p: float = 0.1, rng: np.random.Generator | None = None):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = rng or np.random.default_rng()

    def avant(self, x: np.ndarray) -> np.ndarray:
        if not self.entrainement or self.p == 0.0:
            return x
        masque = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        self.cache["masque"] = masque
        return x * masque

    def arriere(self, grad_sortie: np.ndarray) -> np.ndarray:
        if not self.entrainement or self.p == 0.0:
            return grad_sortie
        return grad_sortie * self.cache["masque"]


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #


class Sequentiel:
    """Empilement de couches, avec passe avant et arrière automatiques.

    Équivalent de `torch.nn.Sequential`. La passe arrière parcourt simplement
    les couches **dans l'ordre inverse** : c'est toute la rétropropagation.
    """

    def __init__(self, *couches: Couche):
        self.couches = list(couches)

    def avant(self, x: np.ndarray) -> np.ndarray:
        for couche in self.couches:
            x = couche.avant(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.avant(x)

    def arriere(self, grad: np.ndarray) -> np.ndarray:
        for couche in reversed(self.couches):
            grad = couche.arriere(grad)
        return grad

    def zero_grad(self) -> None:
        for couche in self.couches:
            couche.zero_grad()

    def parametres(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Liste de couples ``(paramètre, gradient)`` pour l'optimiseur."""
        return [
            (couche.params[nom], couche.grads[nom])
            for couche in self.couches
            for nom in couche.params
        ]

    def nb_parametres(self) -> int:
        return sum(p.size for p, _ in self.parametres())

    def train(self) -> None:
        """Mode entraînement (active le dropout)."""
        for couche in self.couches:
            couche.entrainement = True

    def eval(self) -> None:
        """Mode évaluation (désactive le dropout). Oublier ce passage est un
        classique : les scores de validation deviennent bruités sans raison."""
        for couche in self.couches:
            couche.entrainement = False


# --------------------------------------------------------------------------- #
# Coût
# --------------------------------------------------------------------------- #


class EntropieCroisee:
    """Coût softmax + log-vraisemblance négative, avec son gradient.

    Je fais volontairement du coût un objet séparé des couches : c'est lui qui
    *amorce* la rétropropagation en produisant le premier gradient, celui du
    coût par rapport aux scores de sortie.
    """

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def avant(self, scores: np.ndarray, cibles: np.ndarray) -> float:
        log_p = log_softmax(scores, axe=-1)
        self.cache["scores"] = scores
        self.cache["cibles"] = cibles
        batch = scores.shape[0]
        return float(-np.mean(log_p[np.arange(batch), cibles]))

    def arriere(self) -> np.ndarray:
        """``(softmax(z) - one_hot) / batch`` — cf. la démonstration de la version 1."""
        scores, cibles = self.cache["scores"], self.cache["cibles"]
        batch = scores.shape[0]
        grad = softmax(scores, axe=-1)
        grad[np.arange(batch), cibles] -= 1.0
        return grad / batch


# --------------------------------------------------------------------------- #
# Optimiseurs
# --------------------------------------------------------------------------- #


class SGD:
    """Descente de gradient stochastique, avec inertie (momentum) optionnelle.

    Sans inertie : ``p <- p - lr * grad``.

    Avec inertie : on entretient une moyenne mobile des gradients passés,
    ``v <- mu * v + grad`` puis ``p <- p - lr * v``. Cela lisse la trajectoire
    et accélère nettement dans les « vallées » allongées, où le gradient brut
    oscille d'un flanc à l'autre au lieu de descendre.

    J'ajoute aussi la **décroissance de poids** (weight decay), c'est-à-dire une
    régularisation L2 : on ajoute ``lambda * p`` au gradient, ce qui pousse les
    paramètres vers zéro et limite le surapprentissage.
    """

    def __init__(self, parametres, lr: float = 0.01, inertie: float = 0.0,
                 decroissance_poids: float = 0.0):
        self.parametres = list(parametres)
        self.lr = lr
        self.inertie = inertie
        self.decroissance_poids = decroissance_poids
        self.vitesses = [np.zeros_like(p) for p, _ in self.parametres]

    def pas(self) -> None:
        for i, (p, g) in enumerate(self.parametres):
            grad = g
            if self.decroissance_poids:
                grad = grad + self.decroissance_poids * p
            if self.inertie:
                self.vitesses[i] = self.inertie * self.vitesses[i] + grad
                grad = self.vitesses[i]
            # `-=` sur place : on modifie le tableau NumPy que la couche possède,
            # sans créer de nouvel objet — sinon la couche garderait l'ancien.
            p -= self.lr * grad

    def zero_grad(self) -> None:
        for _, g in self.parametres:
            g.fill(0.0)


class Adam:
    """Adam (Kingma & Ba, 2015) — l'optimiseur par défaut de tous mes TP.

    Adam combine deux idées :

      * un **moment d'ordre 1** ``m`` : moyenne mobile du gradient (l'inertie) ;
      * un **moment d'ordre 2** ``v`` : moyenne mobile du carré du gradient.

    La mise à jour divise ``m`` par ``sqrt(v)``, ce qui donne à chaque paramètre
    son propre pas effectif : les paramètres dont le gradient est
    systématiquement petit avancent quand même. C'est décisif en TAL, où les
    plongements des mots rares reçoivent très peu de signal.

    La **correction de biais** (``/ (1 - beta^t)``) compense le fait que ``m`` et
    ``v``, initialisés à zéro, sous-estiment fortement les vraies moyennes
    pendant les premières itérations.
    """

    def __init__(self, parametres, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 decroissance_poids: float = 0.0):
        self.parametres = list(parametres)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.decroissance_poids = decroissance_poids
        self.m = [np.zeros_like(p) for p, _ in self.parametres]
        self.v = [np.zeros_like(p) for p, _ in self.parametres]
        self.t = 0

    def pas(self) -> None:
        self.t += 1
        for i, (p, g) in enumerate(self.parametres):
            grad = g
            if self.decroissance_poids:
                grad = grad + self.decroissance_poids * p

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_corrige = self.m[i] / (1 - self.beta1 ** self.t)
            v_corrige = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_corrige / (np.sqrt(v_corrige) + self.eps)

    def zero_grad(self) -> None:
        for _, g in self.parametres:
            g.fill(0.0)


# --------------------------------------------------------------------------- #
# Boucle d'entraînement
# --------------------------------------------------------------------------- #


def generer_lots(X: np.ndarray, y: np.ndarray, taille_lot: int,
                 rng: np.random.Generator):
    """Génère des mini-lots dans un ordre aléatoire, à chaque époque.

    Pourquoi mélanger : sans mélange, si le corpus est trié par classe, chaque
    lot ne contient qu'une classe et les mises à jour se contredisent d'un lot
    au suivant. C'est un bug silencieux — la perte descend, mais mal.

    Pourquoi des mini-lots plutôt qu'un exemple à la fois : le gradient d'un lot
    est moins bruité, et surtout un produit matriciel `(32, d) @ (d, h)` est bien
    plus efficace que 32 produits `(1, d) @ (d, h)`.
    """
    n = X.shape[0]
    ordre = rng.permutation(n)
    for debut in range(0, n, taille_lot):
        idx = ordre[debut:debut + taille_lot]
        yield X[idx], y[idx]


def entrainer(
    modele: Sequentiel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    nb_epoques: int = 20,
    taille_lot: int = 32,
    lr: float = 1e-2,
    optimiseur: str = "adam",
    patience: int | None = 5,
    graine: int = 0,
    verbeux: bool = True,
) -> dict[str, list[float]]:
    """Boucle d'entraînement complète, avec arrêt anticipé.

    Les cinq étapes, dans l'ordre — ce sont exactement celles que les énoncés de
    M1 nous faisaient écrire en commentaire avant de coder :

      1. `zero_grad()` — sinon les gradients du lot précédent s'ajoutent ;
      2. passe avant ;
      3. calcul du coût ;
      4. passe arrière ;
      5. mise à jour des paramètres.

    Returns:
        L'historique, pour tracer les courbes d'apprentissage.
    """
    rng = np.random.default_rng(graine)
    cout = EntropieCroisee()
    params = modele.parametres()
    opt = (Adam(params, lr=lr) if optimiseur == "adam"
           else SGD(params, lr=lr, inertie=0.9))

    historique: dict[str, list[float]] = {
        "perte_train": [], "exactitude_train": [],
        "perte_valid": [], "exactitude_valid": [],
    }
    meilleure_perte = float("inf")
    meilleurs_params: list[np.ndarray] | None = None
    epoques_sans_progres = 0

    for epoque in range(nb_epoques):
        modele.train()
        pertes, exactitudes, poids = [], [], []

        for X_lot, y_lot in generer_lots(X_train, y_train, taille_lot, rng):
            modele.zero_grad()                       # 1
            scores = modele.avant(X_lot)             # 2
            perte = cout.avant(scores, y_lot)        # 3
            modele.arriere(cout.arriere())           # 4
            opt.pas()                                # 5

            pertes.append(perte)
            exactitudes.append(float(np.mean(np.argmax(scores, axis=1) == y_lot)))
            poids.append(len(y_lot))

        poids_arr = np.array(poids, dtype=float)
        historique["perte_train"].append(float(np.average(pertes, weights=poids_arr)))
        historique["exactitude_train"].append(
            float(np.average(exactitudes, weights=poids_arr))
        )

        message = (
            f"époque {epoque + 1:>3} | perte {historique['perte_train'][-1]:.4f} "
            f"| exactitude {historique['exactitude_train'][-1]:.4f}"
        )

        if X_valid is not None and y_valid is not None:
            modele.eval()
            scores_v = modele.avant(X_valid)
            perte_v = cout.avant(scores_v, y_valid)
            exact_v = float(np.mean(np.argmax(scores_v, axis=1) == y_valid))
            historique["perte_valid"].append(perte_v)
            historique["exactitude_valid"].append(exact_v)
            message += f" || valid : perte {perte_v:.4f} | exactitude {exact_v:.4f}"

            if perte_v < meilleure_perte - 1e-5:
                meilleure_perte = perte_v
                meilleurs_params = [p.copy() for p, _ in params]
                epoques_sans_progres = 0
            else:
                epoques_sans_progres += 1

        if verbeux:
            print(message)

        if patience is not None and epoques_sans_progres >= patience:
            if verbeux:
                print(f"Arrêt anticipé : {patience} époques sans amélioration.")
            break

    # On restaure le meilleur état vu : c'est *ça*, l'arrêt anticipé. S'arrêter
    # sans restaurer revient à garder le modèle le plus surappris de la série.
    if meilleurs_params is not None:
        for (p, _), meilleur in zip(params, meilleurs_params):
            p[...] = meilleur

    return historique


def courbe_ascii(valeurs: list[float], largeur: int = 60, hauteur: int = 12,
                 titre: str = "") -> str:
    """Petit tracé en caractères, pour visualiser sans matplotlib.

    Les TP produisaient toujours des courbes matplotlib. Comme je veux que ce
    dépôt tourne partout sans dépendance graphique, je trace en ASCII : c'est
    moins joli mais tout aussi lisible dans un terminal ou un log de CI.
    """
    if not valeurs:
        return ""
    vmin, vmax = min(valeurs), max(valeurs)
    etendue = max(vmax - vmin, 1e-12)

    # Ré-échantillonnage sur `largeur` colonnes.
    n = len(valeurs)
    colonnes = [valeurs[int(i * (n - 1) / max(largeur - 1, 1))] for i in range(min(largeur, n))]

    grille = [[" "] * len(colonnes) for _ in range(hauteur)]
    for x, v in enumerate(colonnes):
        y = int((v - vmin) / etendue * (hauteur - 1))
        grille[hauteur - 1 - y][x] = "*"

    lignes = [f"  {titre}" if titre else ""]
    for i, ligne in enumerate(grille):
        valeur_axe = vmax - i * etendue / (hauteur - 1)
        lignes.append(f"  {valeur_axe:8.4f} |{''.join(ligne)}")
    lignes.append("  " + " " * 9 + "+" + "-" * len(colonnes))
    return "\n".join(lignes)


if __name__ == "__main__":
    print("=== MLP entraîné par rétropropagation écrite à la main ===\n")

    rng = np.random.default_rng(0)

    # Problème « en spirale » : classiquement non linéairement séparable, donc
    # impossible pour le perceptron de la version 1, facile pour un MLP.
    def spirales(n_par_classe: int = 300, nb_classes: int = 3):
        X = np.zeros((n_par_classe * nb_classes, 2))
        y = np.zeros(n_par_classe * nb_classes, dtype=int)
        for c in range(nb_classes):
            idx = range(n_par_classe * c, n_par_classe * (c + 1))
            r = np.linspace(0.0, 1.0, n_par_classe)
            t = (np.linspace(c * 4, (c + 1) * 4, n_par_classe)
                 + rng.normal(size=n_par_classe) * 0.2)
            X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[idx] = c
        return X, y

    X, y = spirales()
    melange = rng.permutation(len(y))
    X, y = X[melange], y[melange]
    coupe = int(0.8 * len(y))
    X_train, y_train, X_valid, y_valid = X[:coupe], y[:coupe], X[coupe:], y[coupe:]

    print("--- Modèle linéaire (équivalent au perceptron de la version 1) ---")
    lineaire = Sequentiel(Lineaire(2, 3, rng=rng))
    h_lin = entrainer(lineaire, X_train, y_train, X_valid, y_valid,
                      nb_epoques=40, lr=0.05, verbeux=False, patience=None)
    print(f"exactitude validation : {h_lin['exactitude_valid'][-1]:.4f}")
    print(f"paramètres : {lineaire.nb_parametres()}\n")

    print("--- MLP à une couche cachée ---")
    mlp = Sequentiel(
        Lineaire(2, 64, rng=rng), ReLU(),
        Lineaire(64, 64, rng=rng), ReLU(),
        Lineaire(64, 3, rng=rng),
    )
    h_mlp = entrainer(mlp, X_train, y_train, X_valid, y_valid,
                      nb_epoques=60, lr=0.01, verbeux=False, patience=None)
    print(f"exactitude validation : {h_mlp['exactitude_valid'][-1]:.4f}")
    print(f"paramètres : {mlp.nb_parametres()}\n")

    print(courbe_ascii(h_mlp["perte_train"], titre="perte d'entraînement (MLP)"))
    print()
    print(courbe_ascii(h_mlp["exactitude_valid"], titre="exactitude de validation (MLP)"))

    print(
        "\nLe modèle linéaire plafonne : les spirales ne sont pas linéairement\n"
        "séparables, exactement comme XOR. Le MLP y arrive parce que ses couches\n"
        "cachées déforment l'espace jusqu'à ce que les classes se séparent."
    )
