# Version 2, la rétropropagation, écrite à la main

> **Où j'en suis.** La version 1 posait les briques et montrait pourquoi il faut
> des couches. Ici j'apprends enfin ces couches, je réimplémente en NumPy pur
> le mécanisme complet que PyTorch cache derrière `loss.backward()`.
> Aucune différentiation automatique, aucun `torch`.

---

## Nouveautés par rapport à la version 1

| Fichier | Ce que j'ajoute |
|---|---|
| `src/reseau.py` | Couches `Lineaire`, `ReLU`, `Tanh`, `Dropout` avec passes avant **et arrière**, conteneur `Sequentiel`, coût `EntropieCroisee`, optimiseurs `SGD` (avec inertie) et `Adam`, boucle d'entraînement avec mini-lots et arrêt anticipé, tracé de courbes en ASCII. |
| `src/donnees.py` | Corpus d'identification de la langue (FR/EN/ES) embarqué dans le code, découpage stratifié, sac de n-grammes de caractères. |
| `tests/test_reseau.py` | 11 tests, dont la **vérification par différences finies de tous les paramètres du réseau**. |

Tout ce que contenait la version 1 est conservé à l'identique.

## Origine universitaire

- TP 1 *Introduction to PyTorch* (Timothée Bernard, M2), la partie « recodez
  `nn.Linear`, puis un réseau à une couche cachée sans utiliser les fonctions de
  haut niveau ».
- TP *A first classifier in PyTorch: language identification using a bag of word
  input* (Marie Candito, M1), la tâche, la boucle d'entraînement en cinq
  étapes, l'arrêt anticipé.

## Lancer le code

```bash
python -m src.reseau
```

```bash
python -m src.donnees
```

```bash
python -m tests.test_reseau
```

---

## L'architecture que j'ai choisie, et pourquoi

Chaque couche est un objet qui sait faire deux choses.

```
avant(x)            -> sortie      (et met en cache ce qu'il faudra pour l'arrière)
arriere(dL/dsortie) -> dL/dentrée  (et remplit self.grads)
```

La rétropropagation devient alors une simple boucle sur les couches **dans
l'ordre inverse**. C'est le design des frameworks pré-autograd (Caffe, Torch7),
et il a un immense avantage pédagogique, on voit physiquement le gradient
remonter le réseau, couche après couche.

```
   entrée ──► Linéaire ──► ReLU ──► Linéaire ──► coût
                                                   │
   dL/dx ◄── Linéaire ◄── ReLU ◄── Linéaire ◄──────┘
             (dL/dW,        (masque)   (dL/dW,
              dL/db)                    dL/db)
```

La passe arrière d'une couche linéaire `y = xW + b` donne.

```
dL/dW = xᵀ · dL/dy       dL/db = Σ_batch dL/dy       dL/dx = dL/dy · Wᵀ
```

La transposée n'est pas une astuce d'implémentation, c'est l'adjoint de
l'application linéaire. Une fois qu'on l'a vu une fois, on ne se trompe plus de
sens dans les papiers.

---

## Les trois pièges que je documente dans le code

### 1. `zero_grad()` n'est pas une formalité

J'écris `self.grads["W"] += ...` et non `=`. Les gradients **s'accumulent**,
c'est ce qui permet d'accumuler plusieurs mini-lots avant une mise à jour, mais
c'est aussi ce qui rend `zero_grad()` obligatoire. J'ai écrit un test qui
reproduit le bug volontairement (`test_les_gradients_s_accumulent_sans_zero_grad`)
pour voir concrètement que le gradient double à la seconde passe.

### 2. Le mode `eval()` change réellement le calcul

Le dropout doit être actif à l'entraînement et neutre à l'inférence. J'utilise
la variante *inverted dropout*, on divise les activations survivantes par
`1 - p` à l'entraînement, ce qui rend l'inférence strictement identique à un
réseau sans dropout. Oublier `model.eval()` produit des scores de validation
bruités sans raison apparente, un bug très pénible à diagnostiquer.

### 3. L'arrêt anticipé sans restauration ne sert à rien

S'arrêter après *n* époques sans amélioration puis garder les poids courants,
c'est garder précisément le modèle le plus surappris de la série. Ma boucle
sauvegarde une copie des meilleurs paramètres et les **restaure** à la fin.

---

## Résultats mesurés

**Problème des spirales** (trois classes entrelacées dans le plan, non
linéairement séparables).

| Modèle | Paramètres | Exactitude validation |
|---|---:|---:|
| Linéaire (= perceptron v1) | 9 | 0,59 |
| MLP 2×64 + ReLU | 4 547 | **1,00** |

**Identification de la langue** (FR/EN/ES, 66 phrases, trigrammes de caractères).
exactitude de **1,00** sur le test, avec généralisation correcte sur des phrases
inventées après coup.

**SGD contre Adam** sur un problème volontairement mal conditionné (variables
d'échelles allant de 0,01 à 100). Adam obtient un coût final nettement plus bas.
C'est l'argument concret en faveur d'Adam en TAL, les plongements des mots
rares reçoivent très peu de signal, et un pas unique pour tous les paramètres
les condamne à ne jamais bouger.

---

## Une note méthodologique que je tiens à garder

Dans `src/donnees.py`, le vocabulaire est construit **uniquement sur le train**,
et le découpage train/test est **stratifié** par langue. Ce sont deux réflexes
que je n'avais pas en M1 et qui m'ont valu des scores incompréhensibles.

- vocabulaire construit sur tout le corpus = fuite d'information du test,
- découpage aléatoire global sur un petit corpus = test parfois sans aucun
  exemple d'une classe, donc score ininterprétable.

---

## Ce qui reste ouvert

Mon réseau prend des sacs de mots en entrée, donc des symboles discrets sans
aucune relation entre eux. Pour lui, `chat` est aussi éloigné de `chien` que de
`kilomètre`.

Je ne sais pas encore comment lui donner une représentation où la proximité de
sens serait visible.
