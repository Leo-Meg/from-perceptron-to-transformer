# Version 1, fondations. Algèbre, perceptron, et pourquoi il faut des couches

> **Où j'en suis.** C'est le point de départ. Je ne construis encore aucun réseau
> profond. Je pose les briques numériques et je montre, avec quatre points dans
> le plan, pourquoi on empile des couches.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/algebre.py` | Combinaison linéaire, ReLU et tanh, softmax et log-softmax numériquement stables, entropie croisée et son gradient, similarité cosinus, vérificateur de gradient par différences finies. |
| `src/perceptron.py` | Perceptron multiclasse écrit à la main, règle de Rosenblatt, version moyennée, sac de mots maison. |
| `src/operateurs_logiques.py` | AND, OR et NAND posés à la main, preuve que XOR est hors de portée d'un neurone, MLP à une couche cachée qui le résout. |
| `tests/test_fondations.py` | 13 tests sans dépendance externe, dont la vérification du gradient. |

## Origine universitaire

Ces trois modules sont ma synthèse de trois TP de **Machine Learning for NLP 1**
(Marie Candito, M1 Sciences du langage, Linguistique Informatique, Université
Paris Cité) et du **TP 1 de Machine Learning for NLP 3** (Timothée Bernard, M2).

- *Implementing a multiclass Perceptron for document classification*
- *Neural nets for logical operators*
- *Introduction to PyTorch*, la partie où il faut recoder `nn.Linear` sans
  `nn.Linear`

## Lancer le code

```bash
python -m src.algebre
```

```bash
python -m src.operateurs_logiques
```

```bash
python -m tests.test_fondations
```

NumPy suffit, il n'y a rien d'autre à installer.

---

## Les trois choses que je retiens de cette étape

### 1. La stabilité numérique du softmax n'est pas un détail d'implémentation

`softmax([1000, 1001, 999])` renvoie `[nan, nan, nan]` si on applique la formule
naïvement, parce que `exp(1001)` déborde. Il faut retrancher le maximum avant
l'exponentielle, ce qui est licite puisque le softmax est invariant par
translation. Même chose pour `log_softmax`, qu'il ne faut jamais calculer comme
`log(softmax(x))`.

C'est la raison pour laquelle tous les TP de M1 nous demandaient de faire sortir
au réseau des log-probabilités et d'utiliser `NLLLoss`, plutôt que des
probabilités et une log-vraisemblance calculée à part.

### 2. Le gradient de l'entropie croisée est simple

En composant `log_softmax` et NLL, on obtient

```
dL/dz = (softmax(z) - one_hot(cible)) / batch
```

Ni exponentielle, ni logarithme dans le gradient. C'est pour cette raison que
softmax et entropie croisée vont ensemble. Leur composition est numériquement
stable, alors que chacune prise séparément ne l'est pas. Je le vérifie dans le
code par différences finies, l'écart relatif est de l'ordre de 1e-11.

### 3. La couche cachée change de repère

XOR n'est pas linéairement séparable. Je le vérifie par balayage exhaustif de
68 921 triplets `(w1, w2, b)`, et je rappelle la preuve en deux lignes. Le réseau
à une couche cachée que je pose à la main le résout parce que sa couche cachée
envoie les quatre points dans un espace où ils deviennent séparables.

```
   entrée        représentation cachée        sortie
 (+1, +1)  ->      (+1.00, -1.00)      ->      -1
 (+1, -1)  ->      (+1.00, +1.00)      ->      +1
 (-1, +1)  ->      (+1.00, +1.00)      ->      +1
 (-1, -1)  ->      (-1.00, +1.00)      ->      -1
```

Les deux points à classer `-1` sont ceux qui n'ont pas leurs deux coordonnées
positives après projection, et un simple AND suffit alors.

C'est ce que je retiens pour la suite. Chaque couche d'un réseau profond réécrit
les représentations pour que la tâche devienne plus facile à la couche suivante.

---

## Le constat sur lequel je m'arrête

`matrice_sac_de_mots` produit exactement le même vecteur pour ces deux phrases.

> *le chien mord l'homme*
> *l'homme mord le chien*

Le sac de mots est aveugle à l'ordre. Les architectures qui ont suivi dans
l'histoire du domaine, des plongements à l'attention, n'existent que pour
réintroduire cette information perdue. C'est ce que je veux comprendre en
continuant.

---

## Ce qui reste ouvert

Le réseau qui résout XOR, je l'ai posé à la main. J'ai choisi les poids moi-même
après avoir compris ce qu'il fallait faire. Pour quatre points dans le plan c'est
tenable. Au-delà, non.

Il me manque donc la rétropropagation. C'est le prochain obstacle, et je ne sais
pas encore à quel point il est raide.
