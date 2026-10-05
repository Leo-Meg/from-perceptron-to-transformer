# From the perceptron to the Transformer, in NumPy

Rebuilding the parts of a language model by hand, without PyTorch and without automatic differentiation.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 2.** I am doing this work in stages, each in its
> own folder. I publish as I go rather than once everything is finished.

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

## Published versions

| | Folder | Contents | Tests |
|---|---|---|---:|
| **1** | `1.transformer_python_projet` | Algebra, perceptron, and why layers are needed | 13 |
| **2** | `2.transformer_python_projet` | Back-propagation, written by hand | 11 |

That is **24 tests** in total. Each folder contains everything the previous
one had, plus one step.

---

## Running the latest version

```bash
cd 2.transformer_python_projet
python -m src.reseau
python -m src.donnees
python -m tests.test_reseau
```

---

## What is still open

My network takes bags of words as input, so discrete symbols with no relation
between them. For it, `chat` is as far from `chien` as it is from `kilomètre`.

I do not yet know how to give it a representation in which closeness of meaning
would be visible.

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

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*
