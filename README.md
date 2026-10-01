# From the perceptron to the Transformer, in NumPy

Rebuilding the parts of a language model by hand, without PyTorch and without automatic differentiation.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 1.** This is the first step of work I am doing in
> stages, each in its own folder. Only version 1 exists so far. I publish as I go
> rather than once everything is finished.

---

## Why this repository

Over two years of a Master's in computational linguistics I wrote a lot of
`nn.LSTM(...)`, `nn.MultiheadAttention(...)` and `Trainer(...)`. The lab work ran
and the marks followed. By the end I could not have explained why we divide by
`√d_k` in attention, nor where exactly the gradient of an RNN vanishes.

So I am starting again from the beginning, writing each backward pass by hand and
checking it by finite differences.

The order I follow is historical. At each step I measure a limit, and I stop on
it.

---

## What exists today

### Version 1, foundations. Algebra, perceptron, and why layers are needed

| File | What I do in it |
|---|---|
| `src/algebre.py` | Linear combination, ReLU and tanh, numerically stable softmax and log-softmax, cross-entropy and its gradient, cosine similarity, a finite-difference gradient checker. |
| `src/perceptron.py` | Multiclass perceptron written by hand, Rosenblatt's rule, averaged variant, home-made bag of words. |
| `src/operateurs_logiques.py` | AND, OR and NAND set by hand, a proof that XOR is out of reach for a single neuron, and a one-hidden-layer MLP that solves it. |
| `tests/test_fondations.py` | 13 tests, including the gradient check. |

Academic origin. Three lab assignments from *Machine Learning for NLP 1* (Marie
Candito, M1) and assignment 1 of *Machine Learning for NLP 3* (Timothée Bernard,
M2).

---

## Running the code

```bash
cd 1.transformer_python_projet
python -m src.algebre
python -m src.operateurs_logiques
python -m tests.test_fondations
```

NumPy is enough, there is nothing else to install.

---

## What I take from this step

**The numerical stability of softmax is not an implementation detail.**
`softmax([1000, 1001, 999])` returns `[nan, nan, nan]` with the naive formula,
because `exp(1001)` overflows. You have to subtract the maximum before the
exponential, which is valid since softmax is invariant under translation.

**The cross-entropy gradient is simple.** Composing `log_softmax` with NLL gives
`dL/dz = (softmax(z) - one_hot(target)) / batch`, with no exponential and no
logarithm. I check it by finite differences, the relative error is around 1e-11.

**The hidden layer changes the frame of reference.** XOR is not linearly
separable, which I verify by exhaustively sweeping 68,921 triples. The network
solves it because its hidden layer sends the four points into a space where they
become separable.

---

## What is still open

The network that solves XOR, I set by hand, choosing the weights myself. For four
points in the plane that works. Beyond that, it does not. What I am missing is
back-propagation.

Also, `matrice_sac_de_mots` produces the same vector for *le chien mord l'homme*
and for *l'homme mord le chien*. A bag of words is blind to order.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus is written into the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount. Two
years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my coursework are quoted, not erased.** Where a result I handed in
was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I write down what I found.

**The code is commented in French.**

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*
