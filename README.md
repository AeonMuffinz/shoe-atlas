# Shoe Atlas

**Tell me about this shoe.** Point a model at a product photo and it reads eight things off it — what
kind of shoe it is, the heel height, the closure, the materials, the toe shape and more. Then it does two
useful things with that: it checks a shop's own product data for mistakes, and it finds the same shoe in
other colours.

Built on [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/), 50,025 shoe photos.
The interface is bilingual, Turkish by default.

<!-- GIF 1: upload mode — drop a photo, see the attributes and similar shoes -->

<!-- GIF 2: catalog mode — pick a product, see the audit flag a wrong attribute -->

## What it does

**Reads attributes from a photo.** Eight groups of them. Some are one-of-a-kind — a shoe has exactly one
heel height — and some can have several at once, like materials. The model handles both, and it says how
confident it is.

**Finds mistakes in a shop's product data.** This is the interesting part. Every online shop has products
tagged wrong: the leather boot filed as canvas, the 3-inch heel listed as flat. The model compares what it
sees against what the catalog claims and flags the disagreements.

The hard question is whether those flags are any good, because a model that disagrees with a correct label
is just wrong itself. So the flags are tested properly: take clean data, deliberately corrupt a known
slice of it, retrain, and see how much of the damage gets found. **It finds 54–93% of planted errors**,
depending on the attribute and how badly the data was damaged.

**Finds the same shoe in other colours.** Search the catalog by image. Fine-tuning the model on shoe
attributes made this **2.6× better** than the same network untrained, and it beats CLIP too.

## Try it

You need [uv](https://docs.astral.sh/uv/) and the dataset.

```bash
uv sync
```

```bash
uv run python -m src.prepare_data
```

```bash
uv run python -m src.app
```

That opens the interface. Drop in a shoe photo, or pick a catalog product to see the audit.

To train and score the model yourself:

```bash
uv run python -m src.train --config configs/convnext_base_ar1.yaml
```

```bash
uv run python -m src.evaluate --run artifacts/runs/convnext_base_ar1_s42
```

## How well does it work

The main score is mAP — roughly, "when the model ranks its guesses, how often is the right answer near the
top". Higher is better, 1.0 is perfect.

| | mAP |
|---|---:|
| Guessing from label frequency alone | 0.09 |
| CLIP, no training on shoes | 0.31 |
| Frozen features, only a small layer trained | 0.48 |
| **This model, fully fine-tuned** | **0.57** |

That last row is on the **test set**, held back and untouched until the very end — 0.5745, against 0.5659
on the data used for tuning. It does slightly *better* on data it has never seen, which is the outcome you
want.

Some attributes are much easier than others. Category is 96% correct; heel height is 76%, and most of its
mistakes are one bracket off rather than wild.

## What doesn't work

Kept short and honest.

**"Similar shoes" means "same shoe, different colour."** That is what the data can prove. It is not a
general "looks like this" search, and because colour isn't one of the labelled attributes, colour can't be
evaluated at all.

**The search index includes the training photos.** Right for a demo, but it means the search results
flatter the model a bit. Measured: about a fifth of its advantage is recognising photos it has seen before.

**Every result comes from a single training run.** Re-running with a different random seed would move the
numbers by an unknown amount, because that was never measured. The best model won by a hair — 0.5659 to
0.5633 — and there is no way to say whether that gap is real.

**Half the material labels were dropped** — 31 of 63 — for being too rare to learn, so material
predictions cover a much smaller vocabulary than the raw data suggests. Across all attributes, 44 of 151
labels were dropped.

**Error-finding was only tested on part of the data.** The full cross-validated version, which would let it
audit every product, was not built.

## More detail

Every run, how the model was chosen, what each measurement means and a much longer list of limitations:
**[docs/RESULTS.md](docs/RESULTS.md)**.

## Data and licence

UT Zappos50K, from the University of Texas at Austin, used under its **academic, non-commercial** licence.
Dataset: <https://vision.cs.utexas.edu/projects/finegrained/utzap50k/>

The code here is a university project and comes with no warranty.
