"""
Boîte à outils d'algèbre linéaire — les briques numériques dont tout le reste dépend.

Pourquoi je commence par là
---------------------------
Pendant mon M1, le tout premier TP de Machine Learning (« Introduction to PyTorch »,
Timothée Bernard) consistait à manipuler des tenseurs à la main : produits
matriciels, transpositions, broadcasting, puis à recoder `torch.nn.Linear`
sans utiliser `torch.nn.Linear`. J'ai longtemps trouvé ça un peu scolaire.
J'ai compris plus tard que c'est exactement là que tout se joue : un Transformer
n'est *rien d'autre* qu'un empilement bien choisi de produits matriciels, de
softmax et de normalisations. Si on maîtrise cette page, on peut lire n'importe
quel papier d'architecture.

Je réimplémente donc ici, en NumPy pur, les opérations que j'utiliserai partout
dans le projet. Je les accompagne de vérifications numériques : c'est ma façon
de m'assurer que je n'ai pas seulement recopié une formule, mais que je sais
la tester.

Convention de forme (shape) que je respecte dans tout le dépôt
--------------------------------------------------------------
L'axe 0 est **toujours** l'axe du batch. Un tenseur d'activations a la forme
`(batch, ...)`. C'est la convention de PyTorch avec `batch_first=True`, et c'est
la source d'erreur numéro un quand on débute : je préfère l'écrire noir sur blanc.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# 1. Combinaison linéaire : le cœur d'un réseau de neurones
# --------------------------------------------------------------------------- #


def combinaison_lineaire(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcule ``X @ W + b`` pour un *lot* (batch) de vecteurs d'entrée.

    C'est l'unique opération paramétrique d'une couche dense. Tout le reste
    (activations, softmax, normalisations) n'a pas de paramètre appris.

    Args:
        X: entrées, forme ``(batch, d_entree)``.
        W: matrice de poids, forme ``(d_entree, d_sortie)``.
        b: biais, forme ``(d_sortie,)``.

    Returns:
        Sorties, forme ``(batch, d_sortie)``.

    Note sur le broadcasting:
        ``X @ W`` a la forme ``(batch, d_sortie)`` et ``b`` la forme ``(d_sortie,)``.
        NumPy « diffuse » (broadcast) automatiquement ``b`` sur l'axe du batch :
        le même biais est ajouté à chaque exemple. C'est voulu — le biais est un
        paramètre du modèle, pas de l'exemple.
    """
    assert X.ndim == 2, f"X doit être 2D (batch, d_entree), reçu {X.shape}"
    assert W.ndim == 2, f"W doit être 2D (d_entree, d_sortie), reçu {W.shape}"
    assert X.shape[1] == W.shape[0], (
        f"Dimensions incompatibles : X{X.shape} @ W{W.shape}. "
        "La dimension d'entrée de X doit valoir le nombre de lignes de W."
    )
    assert b.shape == (W.shape[1],), f"b doit avoir la forme {(W.shape[1],)}, reçu {b.shape}"
    return X @ W + b


# --------------------------------------------------------------------------- #
# 2. Fonctions d'activation
# --------------------------------------------------------------------------- #


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU : ``max(0, x)``, appliquée élément par élément.

    C'est l'activation par défaut depuis ~2012. Son intérêt n'est pas sa forme
    (elle est presque triviale) mais sa dérivée : 0 ou 1. Pas d'écrasement du
    gradient comme avec la sigmoïde, donc on peut empiler beaucoup de couches.
    """
    return np.maximum(0.0, x)


def relu_derivee(x: np.ndarray) -> np.ndarray:
    """Dérivée de la ReLU par rapport à son entrée.

    En 0 la fonction n'est pas dérivable ; par convention on renvoie 0.
    En pratique cela n'a aucune conséquence (la probabilité de tomber
    exactement sur 0 est nulle avec des flottants).
    """
    return (x > 0).astype(x.dtype)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tangente hyperbolique, à valeurs dans ]-1, 1[.

    Je l'utilise dans le TP « réseaux pour opérateurs logiques » : elle sature
    vite (tanh(3) > 0.99), ce qui permet de fabriquer *à la main* des neurones
    qui se comportent comme des portes logiques.
    """
    return np.tanh(x)


def tanh_derivee(x: np.ndarray) -> np.ndarray:
    """Dérivée de tanh : ``1 - tanh(x)^2``."""
    t = np.tanh(x)
    return 1.0 - t * t


def signe(x: np.ndarray) -> np.ndarray:
    """Fonction signe utilisée comme activation de sortie binaire (+1 / -1).

    Attention : elle n'est pas dérivable, donc inutilisable en rétropropagation.
    Je m'en sers uniquement pour les réseaux dont je fixe les poids à la main.
    """
    return np.where(x > 0, 1, -1)


# --------------------------------------------------------------------------- #
# 3. Softmax et log-softmax — la stabilité numérique n'est pas un détail
# --------------------------------------------------------------------------- #


def softmax(scores: np.ndarray, axe: int = -1) -> np.ndarray:
    """Transforme des scores réels en distribution de probabilité.

    Le piège classique : ``exp(1000)`` déborde et donne ``inf``. On utilise
    l'invariance du softmax par translation — ``softmax(z) == softmax(z - c)``
    pour tout scalaire ``c`` — en choisissant ``c = max(z)``. Tous les exposants
    deviennent alors ``<= 0``, donc tous les ``exp`` sont dans ``]0, 1]``.

    Args:
        scores: tenseur de scores (« logits » au sens large).
        axe: axe sur lequel normaliser. Se tromper d'axe est l'erreur la plus
            fréquente et la plus silencieuse : le modèle « apprend » quand même,
            mais mal.
    """
    scores_stables = scores - np.max(scores, axis=axe, keepdims=True)
    exp = np.exp(scores_stables)
    return exp / np.sum(exp, axis=axe, keepdims=True)


def log_softmax(scores: np.ndarray, axe: int = -1) -> np.ndarray:
    """Logarithme du softmax, calculé sans passer par ``log(softmax(...))``.

    Pourquoi : si une probabilité vaut 1e-40, ``softmax`` la stocke comme 0.0
    et ``log(0)`` vaut ``-inf``. En calculant directement
    ``z - max(z) - log(sum(exp(z - max(z))))`` on ne perd jamais la précision.

    C'est exactement ce que fait ``torch.nn.functional.log_softmax``, et c'est
    la raison pour laquelle les TP de M1 nous demandaient toujours de sortir des
    log-probabilités et d'utiliser ``NLLLoss`` plutôt que ``softmax`` + ``log``.
    """
    scores_stables = scores - np.max(scores, axis=axe, keepdims=True)
    return scores_stables - np.log(np.sum(np.exp(scores_stables), axis=axe, keepdims=True))


# --------------------------------------------------------------------------- #
# 4. Fonctions de coût
# --------------------------------------------------------------------------- #


def perte_log_vraisemblance_negative(log_probas: np.ndarray, cibles: np.ndarray) -> float:
    """NLL (Negative Log-Likelihood) moyennée sur le lot.

    Args:
        log_probas: forme ``(batch, nb_classes)``, sorties d'un ``log_softmax``.
        cibles: forme ``(batch,)``, indices entiers des classes correctes.

    Returns:
        Le coût moyen (un scalaire).

    Intuition: on ne regarde qu'une seule case par ligne — celle de la bonne
    classe — et on veut que sa log-probabilité soit la plus proche possible de 0
    (c'est-à-dire une probabilité proche de 1). Le signe moins transforme
    « maximiser la vraisemblance » en « minimiser un coût ».
    """
    batch = log_probas.shape[0]
    return float(-np.mean(log_probas[np.arange(batch), cibles]))


def perte_entropie_croisee(scores: np.ndarray, cibles: np.ndarray) -> float:
    """Entropie croisée = ``log_softmax`` puis ``NLL``, en une seule fonction.

    C'est l'équivalent de ``torch.nn.CrossEntropyLoss``, qui prend des *scores
    bruts* et non des probabilités. Confondre les deux (appliquer un softmax
    puis CrossEntropyLoss) est un bug classique qui ne fait pas planter le code
    mais dégrade discrètement l'apprentissage.
    """
    return perte_log_vraisemblance_negative(log_softmax(scores, axe=-1), cibles)


def gradient_entropie_croisee(scores: np.ndarray, cibles: np.ndarray) -> np.ndarray:
    """Gradient de l'entropie croisée par rapport aux **scores** d'entrée.

    Le résultat est d'une simplicité remarquable :

        dL/dz = (softmax(z) - one_hot(cible)) / batch

    C'est le premier résultat qui m'a fait comprendre pourquoi l'entropie
    croisée et le softmax vont toujours ensemble : leur composition a un
    gradient qui ne contient ni exponentielle ni logarithme, donc numériquement
    parfaitement stable. Toutes les autres combinaisons sont plus fragiles.
    """
    batch = scores.shape[0]
    probas = softmax(scores, axe=-1)
    probas[np.arange(batch), cibles] -= 1.0
    return probas / batch


# --------------------------------------------------------------------------- #
# 5. Similarité — indispensable dès qu'on parle de plongements (embeddings)
# --------------------------------------------------------------------------- #


def similarite_cosinus(a: np.ndarray, b: np.ndarray) -> float:
    """Cosinus de l'angle entre deux vecteurs.

    En TAL on préfère presque toujours le cosinus à la distance euclidienne :
    il ignore la norme des vecteurs et ne compare que leur *direction*. Or la
    norme d'un plongement encode surtout la fréquence du mot, pas son sens.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def matrice_similarite_cosinus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Toutes les similarités cosinus entre les lignes de ``A`` et celles de ``B``.

    Version vectorisée : on normalise chaque ligne puis on fait un seul produit
    matriciel. Sur un vocabulaire de 30 000 mots, la boucle Python est
    impraticable alors que cette version tient en quelques millisecondes.

    Returns:
        Forme ``(n_a, n_b)``.
    """
    A_norm = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B_norm = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return A_norm @ B_norm.T


# --------------------------------------------------------------------------- #
# 6. Vérification de gradient — mon garde-fou pour tout le projet
# --------------------------------------------------------------------------- #


def verifier_gradient(
    fonction,
    gradient_analytique,
    x: np.ndarray,
    epsilon: float = 1e-5,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Compare un gradient calculé à la main à une approximation par différences finies.

    Principe : pour chaque coordonnée ``i``, on approche

        dF/dx_i ≈ (F(x + eps.e_i) - F(x - eps.e_i)) / (2.eps)

    (différence centrée, erreur en O(eps²) au lieu de O(eps) pour la différence
    à droite). Si mon gradient analytique s'écarte trop de cette approximation,
    c'est que j'ai fait une erreur de dérivation.

    C'est LA technique qui m'a évité des heures de débogage : quand un réseau
    « n'apprend pas », dans 80 % des cas c'est une erreur de backward, pas un
    problème d'hyperparamètre.

    Returns:
        ``(succes, ecart_relatif_max)``.
    """
    grad_analytique = gradient_analytique(x)
    grad_numerique = np.zeros_like(x, dtype=float)

    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        valeur = x[idx]

        x[idx] = valeur + epsilon
        f_plus = fonction(x)
        x[idx] = valeur - epsilon
        f_moins = fonction(x)
        x[idx] = valeur  # on restaure : la fonction ne doit rien casser

        grad_numerique[idx] = (f_plus - f_moins) / (2 * epsilon)
        it.iternext()

    denominateur = np.maximum(
        np.abs(grad_analytique) + np.abs(grad_numerique), 1e-12
    )
    ecart = float(np.max(np.abs(grad_analytique - grad_numerique) / denominateur))
    return ecart < tolerance, ecart


if __name__ == "__main__":
    print("=== Démonstration rapide de la boîte à outils ===\n")

    rng = np.random.default_rng(42)

    X = rng.random((4, 10))
    W = rng.random((10, 3))
    b = rng.random(3)
    Z = combinaison_lineaire(X, W, b)
    print(f"X{X.shape} @ W{W.shape} + b{b.shape} -> {Z.shape}")
    print(f"Classes prédites (argmax) : {np.argmax(Z, axis=1)}\n")

    # Le softmax résiste-t-il à des scores énormes ?
    scores_extremes = np.array([[1000.0, 1001.0, 999.0]])
    print(f"softmax de scores énormes : {softmax(scores_extremes)}")
    print("  (sans l'astuce du max, on aurait obtenu [nan nan nan])\n")

    # La somme d'une distribution vaut bien 1
    assert np.allclose(softmax(Z).sum(axis=1), 1.0)
    # log_softmax et log(softmax) coïncident dans le régime « facile »
    assert np.allclose(log_softmax(Z), np.log(softmax(Z)))
    print("Contrôles softmax / log_softmax : OK\n")

    # Vérification du gradient de l'entropie croisée
    cibles = np.array([0, 1, 2, 1])
    ok, ecart = verifier_gradient(
        fonction=lambda z: perte_entropie_croisee(z, cibles),
        gradient_analytique=lambda z: gradient_entropie_croisee(z, cibles),
        x=Z.copy(),
        tolerance=1e-5,
    )
    print(f"Gradient de l'entropie croisée vérifié : {ok} (écart relatif max = {ecart:.2e})")
