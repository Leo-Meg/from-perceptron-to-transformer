"""
Tests de la version 1 — exécutables sans aucune dépendance hormis NumPy.

    python -m tests.test_fondations

J'ai pris l'habitude d'écrire ces tests après un TP de M2 où j'avais passé une
soirée à chercher pourquoi un modèle n'apprenait pas : le bug était dans mon
`softmax`, appliqué sur le mauvais axe. Depuis, je teste chaque brique isolément
avant de l'assembler.
"""

from __future__ import annotations

import numpy as np

from src.algebre import (
    combinaison_lineaire,
    gradient_entropie_croisee,
    log_softmax,
    matrice_similarite_cosinus,
    perte_entropie_croisee,
    relu,
    relu_derivee,
    similarite_cosinus,
    softmax,
    verifier_gradient,
)
from src.operateurs_logiques import (
    CIBLES,
    ENTREES,
    PORTE_AND,
    PORTE_NAND,
    PORTE_OR,
    ReseauXOR,
)
from src.perceptron import PerceptronMulticlasse, matrice_sac_de_mots


def test_combinaison_lineaire() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([10.0, 20.0])
    attendu = np.array([[11.0, 22.0], [13.0, 24.0]])
    assert np.allclose(combinaison_lineaire(X, W, b), attendu)


def test_softmax_est_une_distribution() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(7, 5)) * 10
    p = softmax(scores)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0)


def test_softmax_stable_sur_grands_scores() -> None:
    """Sans le retrait du maximum, ce cas produirait des NaN."""
    scores = np.array([[10_000.0, 10_001.0, 9_999.0]])
    p = softmax(scores)
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)


def test_log_softmax_coherent_avec_softmax() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(4, 6))
    assert np.allclose(log_softmax(scores), np.log(softmax(scores)))


def test_relu_et_sa_derivee() -> None:
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.allclose(relu(x), [0.0, 0.0, 0.0, 0.5, 2.0])
    assert np.allclose(relu_derivee(x), [0.0, 0.0, 0.0, 1.0, 1.0])


def test_gradient_entropie_croisee() -> None:
    """Le test le plus important du fichier : dérivation manuelle vs différences finies."""
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(5, 4))
    cibles = rng.integers(0, 4, size=5)
    ok, ecart = verifier_gradient(
        fonction=lambda z: perte_entropie_croisee(z, cibles),
        gradient_analytique=lambda z: gradient_entropie_croisee(z, cibles),
        x=scores.copy(),
        tolerance=1e-5,
    )
    assert ok, f"gradient incorrect, écart relatif = {ecart:.2e}"


def test_similarite_cosinus() -> None:
    a = np.array([1.0, 0.0])
    assert np.isclose(similarite_cosinus(a, np.array([2.0, 0.0])), 1.0)   # même direction
    assert np.isclose(similarite_cosinus(a, np.array([0.0, 1.0])), 0.0)   # orthogonaux
    assert np.isclose(similarite_cosinus(a, np.array([-1.0, 0.0])), -1.0)  # opposés

    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    M = matrice_similarite_cosinus(A, A)
    assert np.allclose(M, np.eye(2))


def test_portes_logiques_a_un_neurone() -> None:
    for porte in (PORTE_AND, PORTE_OR, PORTE_NAND):
        assert np.all(porte(ENTREES) == CIBLES[porte.nom]), f"{porte.nom} incorrecte"


def test_xor_necessite_une_couche_cachee() -> None:
    # Aucun neurone linéaire ne résout XOR : je le vérifie sur les poids qui
    # marchent pour les autres portes.
    for porte in (PORTE_AND, PORTE_OR, PORTE_NAND):
        assert not np.all(porte(ENTREES) == CIBLES["XOR"])
    # Le réseau à une couche cachée, lui, y arrive.
    assert np.all(ReseauXOR()(ENTREES) == CIBLES["XOR"])


def test_marge_invariante_par_changement_echelle() -> None:
    from src.operateurs_logiques import NeuroneLineaire

    petit = NeuroneLineaire([1, 1], -1)
    grand = NeuroneLineaire([100, 100], -100)
    assert np.isclose(
        petit.marge(ENTREES, CIBLES["AND"]), grand.marge(ENTREES, CIBLES["AND"])
    )


def test_perceptron_apprend_donnees_separables() -> None:
    rng = np.random.default_rng(3)
    prototypes = rng.normal(size=(3, 20)) * 4.0
    y = rng.integers(0, 3, size=300)
    X = prototypes[y] + rng.normal(size=(300, 20)) * 0.5

    modele = PerceptronMulticlasse(nb_classes=3, moyenne=True)
    modele.entrainer(X, y, nb_epoques=5, verbeux=False)
    assert modele.evaluer(X, y) > 0.95


def test_sac_de_mots_perd_l_ordre() -> None:
    """Le constat qui motive tout le reste du projet."""
    X, _ = matrice_sac_de_mots(
        [
            "le chien mord l' homme".split(),
            "l' homme mord le chien".split(),
        ]
    )
    assert np.allclose(X[0], X[1]), "le sac de mots devrait être aveugle à l'ordre"


def test_sac_de_mots_ignore_mots_inconnus() -> None:
    _, vocab = matrice_sac_de_mots([["a", "b"]])
    X, _ = matrice_sac_de_mots([["a", "zzz"]], vocabulaire=vocab)
    assert X.shape == (1, 2)
    assert np.allclose(X[0], [1.0, 0.0])


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
    print("=== Tests — version 1 : fondations ===\n")
    executer_tous_les_tests()
