"""
Tests de la version 2 — rétropropagation, optimiseurs, boucle d'entraînement.

    python -m tests.test_reseau

Le test le plus important est `test_gradient_du_reseau_complet` : il compare
mon `arriere()` à une approximation par différences finies, paramètre par
paramètre. Si celui-là passe, ma rétropropagation est correcte. C'est le test
que j'aurais dû écrire dès mes premiers TP.
"""

from __future__ import annotations

import numpy as np

from src.donnees import corpus_identification_langue, ngrammes_caracteres
from src.reseau import (
    Adam,
    Dropout,
    EntropieCroisee,
    Lineaire,
    ReLU,
    SGD,
    Sequentiel,
    Tanh,
    entrainer,
    generer_lots,
)


def _reseau_jouet(rng: np.random.Generator) -> Sequentiel:
    return Sequentiel(
        Lineaire(5, 7, rng=rng), Tanh(),
        Lineaire(7, 4, rng=rng), ReLU(),
        Lineaire(4, 3, rng=rng),
    )


def test_formes_passe_avant() -> None:
    rng = np.random.default_rng(0)
    modele = _reseau_jouet(rng)
    sortie = modele.avant(rng.normal(size=(11, 5)))
    assert sortie.shape == (11, 3)


def test_gradient_du_reseau_complet() -> None:
    """Différences finies sur *chaque* paramètre du réseau.

    Méthode : je perturbe une coordonnée d'un paramètre de ±eps, je recalcule le
    coût complet, et je compare la pente obtenue au gradient que ma passe
    arrière a produit. Une erreur de transposée ou d'axe se voit immédiatement.
    """
    rng = np.random.default_rng(1)
    modele = _reseau_jouet(rng)
    cout = EntropieCroisee()

    X = rng.normal(size=(6, 5))
    y = rng.integers(0, 3, size=6)

    modele.zero_grad()
    cout.avant(modele.avant(X), y)
    modele.arriere(cout.arriere())

    eps = 1e-6
    for parametre, gradient in modele.parametres():
        # J'échantillonne quelques coordonnées : tester toutes les coordonnées
        # de tous les paramètres serait correct mais inutilement lent.
        indices = [
            tuple(rng.integers(0, d) for d in parametre.shape) for _ in range(8)
        ]
        for idx in indices:
            initial = parametre[idx]

            parametre[idx] = initial + eps
            cout_plus = cout.avant(modele.avant(X), y)
            parametre[idx] = initial - eps
            cout_moins = cout.avant(modele.avant(X), y)
            parametre[idx] = initial

            pente_numerique = (cout_plus - cout_moins) / (2 * eps)
            pente_analytique = gradient[idx]
            denominateur = max(abs(pente_numerique) + abs(pente_analytique), 1e-9)
            ecart = abs(pente_numerique - pente_analytique) / denominateur
            assert ecart < 1e-4, (
                f"gradient faux en {idx} : analytique {pente_analytique:.6e} "
                f"vs numérique {pente_numerique:.6e} (écart relatif {ecart:.2e})"
            )


def test_les_gradients_s_accumulent_sans_zero_grad() -> None:
    """Reproduction volontaire du bug que `zero_grad()` prévient."""
    rng = np.random.default_rng(2)
    modele = Sequentiel(Lineaire(4, 3, rng=rng))
    cout = EntropieCroisee()
    X, y = rng.normal(size=(5, 4)), rng.integers(0, 3, size=5)

    modele.zero_grad()
    cout.avant(modele.avant(X), y)
    modele.arriere(cout.arriere())
    apres_une_passe = modele.couches[0].grads["W"].copy()

    # Deuxième passe SANS remise à zéro : le gradient double.
    cout.avant(modele.avant(X), y)
    modele.arriere(cout.arriere())
    apres_deux_passes = modele.couches[0].grads["W"]

    assert np.allclose(apres_deux_passes, 2 * apres_une_passe)

    modele.zero_grad()
    assert np.allclose(modele.couches[0].grads["W"], 0.0)


def test_dropout_actif_seulement_en_entrainement() -> None:
    rng = np.random.default_rng(3)
    d = Dropout(p=0.5, rng=rng)
    x = np.ones((100, 100))

    d.entrainement = True
    sortie_train = d.avant(x)
    assert np.any(sortie_train == 0.0), "le dropout devrait éteindre des unités"
    # « Inverted dropout » : l'espérance est préservée.
    assert abs(sortie_train.mean() - 1.0) < 0.05

    d.entrainement = False
    assert np.allclose(d.avant(x), x), "à l'inférence, le dropout doit être neutre"


def test_mode_eval_desactive_le_dropout() -> None:
    rng = np.random.default_rng(4)
    modele = Sequentiel(Lineaire(4, 8, rng=rng), ReLU(), Dropout(0.5, rng=rng),
                        Lineaire(8, 2, rng=rng))
    X = rng.normal(size=(20, 4))
    modele.eval()
    assert np.allclose(modele.avant(X), modele.avant(X)), (
        "en mode eval, deux passes doivent donner exactement le même résultat"
    )


def test_generer_lots_couvre_tout_le_corpus() -> None:
    rng = np.random.default_rng(5)
    X = np.arange(100).reshape(100, 1).astype(float)
    y = np.arange(100)
    vus = []
    for _, y_lot in generer_lots(X, y, taille_lot=7, rng=rng):
        vus.extend(y_lot.tolist())
    assert sorted(vus) == list(range(100)), "chaque exemple doit être vu une fois"


def test_sgd_diminue_le_cout() -> None:
    rng = np.random.default_rng(6)
    modele = Sequentiel(Lineaire(4, 3, rng=rng))
    cout = EntropieCroisee()
    opt = SGD(modele.parametres(), lr=0.1)
    X, y = rng.normal(size=(32, 4)), rng.integers(0, 3, size=32)

    premier = cout.avant(modele.avant(X), y)
    for _ in range(50):
        modele.zero_grad()
        cout.avant(modele.avant(X), y)
        modele.arriere(cout.arriere())
        opt.pas()
    dernier = cout.avant(modele.avant(X), y)
    assert dernier < premier


def test_adam_converge_plus_vite_que_sgd_ici() -> None:
    """Sur un problème mal conditionné, Adam prend l'avantage.

    Je fabrique volontairement des variables d'échelles très différentes : c'est
    la situation où le pas unique de SGD est forcément un mauvais compromis.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 6)) * np.array([1.0, 100.0, 1.0, 0.01, 1.0, 50.0])
    y = (X[:, 1] > 0).astype(int)

    resultats = {}
    for nom, classe in (("sgd", SGD), ("adam", Adam)):
        rng_local = np.random.default_rng(0)
        modele = Sequentiel(Lineaire(6, 8, rng=rng_local), ReLU(),
                            Lineaire(8, 2, rng=rng_local))
        cout = EntropieCroisee()
        opt = classe(modele.parametres(), lr=0.01)
        for _ in range(200):
            modele.zero_grad()
            cout.avant(modele.avant(X), y)
            modele.arriere(cout.arriere())
            opt.pas()
        resultats[nom] = cout.avant(modele.avant(X), y)

    assert resultats["adam"] < resultats["sgd"], resultats


def test_mlp_bat_le_lineaire_sur_probleme_non_lineaire() -> None:
    """Le XOR de la version 1, en version continue et apprise cette fois."""
    rng = np.random.default_rng(8)
    X = rng.normal(size=(600, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # XOR sur les signes

    lineaire = Sequentiel(Lineaire(2, 2, rng=np.random.default_rng(0)))
    entrainer(lineaire, X, y, nb_epoques=40, lr=0.05, patience=None, verbeux=False)
    acc_lineaire = float(np.mean(np.argmax(lineaire.avant(X), axis=1) == y))

    mlp = Sequentiel(
        Lineaire(2, 32, rng=np.random.default_rng(0)), ReLU(),
        Lineaire(32, 2, rng=np.random.default_rng(1)),
    )
    entrainer(mlp, X, y, nb_epoques=80, lr=0.02, patience=None, verbeux=False)
    acc_mlp = float(np.mean(np.argmax(mlp.avant(X), axis=1) == y))

    assert acc_lineaire < 0.75, f"un modèle linéaire ne devrait pas résoudre XOR ({acc_lineaire})"
    assert acc_mlp > 0.90, f"le MLP devrait y arriver ({acc_mlp})"


def test_identification_langue() -> None:
    X_train, y_train, X_test, y_test, vocab, langues = corpus_identification_langue(n=3)
    assert len(langues) == 3
    assert X_train.shape[1] == len(vocab)
    # Chaque langue est représentée dans le test (grâce au découpage stratifié).
    assert set(y_test.tolist()) == {0, 1, 2}

    modele = Sequentiel(
        Lineaire(X_train.shape[1], 32, rng=np.random.default_rng(0)), ReLU(),
        Lineaire(32, 3, rng=np.random.default_rng(1)),
    )
    entrainer(modele, X_train, y_train, nb_epoques=200, taille_lot=8,
              lr=0.01, patience=None, verbeux=False)
    modele.eval()
    exactitude = float(np.mean(np.argmax(modele.avant(X_test), axis=1) == y_test))
    assert exactitude > 0.8, exactitude


def test_ngrammes_caracteres() -> None:
    assert ngrammes_caracteres("ab c", 2) == ["_a", "ab", "b_", "_c", "c_"]


def executer_tous_les_tests() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    echecs = 0
    for test in tests:
        try:
            test()
            print(f"  [OK]     {test.__name__}")
        except AssertionError as e:
            echecs += 1
            print(f"  [ÉCHEC]  {test.__name__} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests passés.")
    if echecs:
        raise SystemExit(1)


if __name__ == "__main__":
    print("=== Tests — version 2 : rétropropagation ===\n")
    executer_tous_les_tests()
