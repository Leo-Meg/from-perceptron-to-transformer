# From the perceptron to the Transformer, in NumPy

Rebuilding the parts of a language model by hand, without PyTorch and without automatic differentiation.

**Léo Mégret** — implementation, French technical notes.  
**Haeeul Hwang** — English documentation, Korean documentation, review.

> **Repository status** — step 2 of 6, landed by Léo Mégret.

---

## Why this repository

Over two years of a Master's in computational linguistics Léo wrote a lot of
`nn.LSTM(...)`, `nn.MultiheadAttention(...)` and `Trainer(...)`. The lab work ran
and the marks followed. By the end he could not have explained why we divide by
`√d_k` in attention, nor where exactly the gradient of an RNN vanishes.

So this repository starts again from the beginning, writing each backward pass by
hand and checking it by finite differences.

The order is historical. At each step we measure a limit, and we stop on it.

---

## Steps

| | Step | Modules | Tests | Landed by | Detailed note (French) |
|---|---|---|---:|---|---|
| **1** | Algebra, the perceptron, and why layers are needed | `src/algebre.py`, `src/perceptron.py`, `src/operateurs_logiques.py` | 13 | Léo | [`algebre-et-perceptron`](notes/algebre-et-perceptron.md) |
| **2** | Back-propagation, written by hand | `src/reseau.py`, `src/donnees.py` | 11 | Léo | [`retropropagation`](notes/retropropagation.md) |

**24** tests in total.

---

## What each step found

**1. Algebra, the perceptron, and why layers are needed**  
A bag of words gives the exact same vector to *the dog bites the man* and *the man bites the dog*. Every architecture that follows exists to put that lost information back.

**2. Back-propagation, written by hand**  
On the spirals problem a linear model is stuck at chance; a two-layer network separates the classes. Every gradient is checked against finite differences.

---

## Running it

```bash
python -m src.reseau
python -m src.donnees

python -m tests.test_fondations
python -m tests.test_reseau
```

---

## How we work

**Nothing to download.** The corpus is written into the code. All of
Léo's Master's notebooks began with a `wget` to a university server or a Google
Drive mount. Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in the original coursework are quoted, not erased.** Where a result
that was handed in was wrong or incomplete, we say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what we
expected, we write down what we found.

**The code is commented in French**, because that is the language it was thought
out in. The documentation is in English, and in Korean where Haeeul has caught
up.

---

*Français : [README_FR.md](README_FR.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
