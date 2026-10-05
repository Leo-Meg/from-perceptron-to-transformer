# Du perceptron au Transformer, en NumPy

Reconstruire les briques d'un modèle de langue à la main, sans PyTorch et sans différentiation automatique.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 2.** Je mène ce travail par étapes, chacune dans son
> propre dossier. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

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

## Les versions publiées

| | Dossier | Contenu | Tests |
|---|---|---|---:|
| **1** | `1.transformer_python_projet` | Algèbre, perceptron, et pourquoi il faut des couches | 13 |
| **2** | `2.transformer_python_projet` | La rétropropagation, écrite à la main | 11 |

Soit **24 tests** au total. Chaque dossier contient tout le contenu du
précédent, plus une étape.

---

## Lancer la dernière version

```bash
cd 2.transformer_python_projet
python -m src.reseau
python -m src.donnees
python -m tests.test_reseau
```

---

## Ce qui reste ouvert

Mon réseau prend des sacs de mots en entrée, donc des symboles discrets sans
aucune relation entre eux. Pour lui, `chat` est aussi éloigné de `chien` que de
`kilomètre`.

Je ne sais pas encore comment lui donner une représentation où la proximité de
sens serait visible.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est écrit dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive. Deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français.**

---

---

*Version anglaise, [README.md](README.md).*
