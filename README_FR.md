# Du perceptron au Transformer, en NumPy

Reconstruire les briques d'un modèle de langue à la main, sans PyTorch et sans différentiation automatique.

**Léo Mégret** — implémentation, notes techniques en français.  
**Haeeul Hwang** — documentation anglaise, documentation coréenne, relecture.

> **État du dépôt** — étape 2 sur 6, publiée par Léo Mégret.

---

## Pourquoi ce dépôt

Pendant deux ans de master en linguistique informatique, j'ai écrit beaucoup de
`nn.LSTM(...)`, de `nn.MultiheadAttention(...)` et de `Trainer(...)`. Les TP
fonctionnaient et les notes suivaient. À la fin je n'aurais pas su expliquer
pourquoi on divise par `√d_k` dans l'attention, ni où exactement le gradient d'un
RNN disparaît.

Je reprends donc depuis le début, en écrivant chaque passe arrière à la main et
en la vérifiant par différences finies.

L'ordre que je suis est historique. À chaque étape je mesure une limite, et je
m'arrête dessus.

---

## Les étapes

| | Étape | Modules | Tests | Publiée par | Note détaillée (français) |
|---|---|---|---:|---|---|
| **1** | Algèbre, perceptron, et pourquoi il faut des couches | `src/algebre.py`, `src/perceptron.py`, `src/operateurs_logiques.py` | 13 | Léo | [`algebre-et-perceptron`](notes/algebre-et-perceptron.md) |
| **2** | La rétropropagation, écrite à la main | `src/reseau.py`, `src/donnees.py` | 11 | Léo | [`retropropagation`](notes/retropropagation.md) |

**24** tests au total.

---

## Ce que chaque étape a trouvé

**1. Algèbre, perceptron, et pourquoi il faut des couches**  
Le sac de mots donne exactement le même vecteur à *le chien mord l'homme* et à *l'homme mord le chien*. Toutes les architectures qui suivent n'existent que pour réintroduire cette information perdue.

**2. La rétropropagation, écrite à la main**  
Sur le problème des spirales, un modèle linéaire reste au niveau du hasard ; deux couches séparent les classes. Chaque gradient est vérifié par différences finies.

---

## Lancer le code

```bash
python -m src.reseau
python -m src.donnees

python -m tests.test_fondations
python -m tests.test_reseau
```

---

## Comment nous travaillons

**Rien à télécharger.** Le corpus est écrit dans le code. Tous mes
notebooks de master commençaient par un `wget` vers un serveur universitaire ou
un montage de Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à
une commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu était faux ou incomplet, je le dis et je donne le bon.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français**, parce que c'est la langue dans laquelle je
l'ai pensé. La documentation est en anglais, et en coréen quand Haeeul est
passée dessus.

---

*English: [README.md](README.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
