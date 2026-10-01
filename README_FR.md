# Du perceptron au Transformer, en NumPy

Reconstruire les briques d'un modèle de langue à la main, sans PyTorch et sans différentiation automatique.

**Léo Mégret** — Master Linguistique Informatique, Université Paris Cité

> **État du dépôt : version 1.** C'est la première étape d'un travail que je
> mène par étapes, chacune dans son propre dossier. Seule la version 1 existe à
> ce jour. Je publie au fur et à mesure plutôt qu'une fois tout terminé, parce
> que l'intérêt de ce travail est justement l'enchaînement des questions.

---

## Pourquoi ce dépôt

Pendant deux ans de master en linguistique informatique, j'ai écrit beaucoup de
`nn.LSTM(...)`, de `nn.MultiheadAttention(...)`, de `Trainer(...)`. Les TP
fonctionnaient, les notes suivaient. Et pourtant, à la fin, je n'aurais pas su
expliquer pourquoi on divise par `√d_k` dans l'attention, ni où exactement le
gradient d'un RNN disparaît.

Ce dépôt est ma réponse à ça. Je reprends depuis le début, en écrivant chaque
passe arrière à la main et en la vérifiant par différences finies.

L'ordre que je suis n'est pas thématique, il est historique : à chaque étape je
mesure une limite, et je m'arrête dessus.

---

## Ce qui existe aujourd'hui

### Version 1 — Fondations : algèbre, perceptron, et pourquoi il faut des couches

| Fichier | Ce que j'y fais |
|---|---|
| `src/algebre.py` | Combinaison linéaire, ReLU et tanh, softmax et log-softmax numériquement stables, entropie croisée et son gradient, similarité cosinus, vérificateur de gradient par différences finies. |
| `src/perceptron.py` | Perceptron multiclasse écrit à la main, règle de Rosenblatt, version moyennée, sac de mots maison. |
| `src/operateurs_logiques.py` | AND, OR et NAND posés à la main, preuve que XOR est hors de portée d'un neurone, MLP à une couche cachée qui le résout. |
| `tests/test_fondations.py` | 13 tests, dont la vérification du gradient. |

Origine universitaire : trois TP de *Machine Learning for NLP 1* (Marie Candito,
M1) et le TP 1 de *Machine Learning for NLP 3* (Timothée Bernard, M2).

---

## Lancer le code

```bash
cd 1.transformer_python_projet
python -m src.algebre
python -m src.operateurs_logiques
python -m tests.test_fondations
```

NumPy suffit. Aucune autre installation.

---

## Ce que je retiens de cette étape

**La stabilité numérique du softmax n'est pas un détail d'implémentation.**
`softmax([1000, 1001, 999])` renvoie `[nan, nan, nan]` avec la formule naïve.

**Le gradient de l'entropie croisée est d'une simplicité qui n'est pas un
hasard.** En composant `log_softmax` et NLL on obtient
`dL/dz = (softmax(z) - one_hot(cible)) / batch`, sans exponentielle ni
logarithme. Vérifié par différences finies, écart relatif de l'ordre de 1e-11.

**La couche cachée ne comprend rien, elle change de repère.** XOR n'est pas
linéairement séparable, je le vérifie par balayage exhaustif de 68 921 triplets.
Le réseau le résout parce que sa couche cachée envoie les quatre points dans un
espace où ils deviennent séparables.

---

## Ce qui reste ouvert

Le réseau qui résout XOR, je l'ai posé à la main : j'ai choisi les poids
moi-même. Pour quatre points dans le plan c'est tenable, au-delà non. Il me
manque la rétropropagation.

Et `matrice_sac_de_mots` produit exactement le même vecteur pour *le chien mord
l'homme* et *l'homme mord le chien*. Le sac de mots est aveugle à l'ordre.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, je change la conclusion, pas l'expérience.

**Le code est commenté en français.** C'est un dépôt à lire autant qu'à exécuter.

---

*Version anglaise : [README.md](README.md).*
