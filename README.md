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

**Finds mistakes in a shop's product data.** Every online shop has products tagged wrong: the leather boot
filed as canvas, the 3-inch heel listed as flat. The model compares what it sees against what the catalog
claims and flags the disagreements.

Those flags are tested rather than trusted. Clean data is deliberately corrupted by a known amount, the
model is retrained on it, and the audit is scored on how much of the planted damage it finds.

**Finds the same shoe in other colours.** Search the catalog by image. Fine-tuning on shoe attributes made
this 2.6× better than the same network untrained, and it beats CLIP.

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

## Results and method

Every run, how the model was selected, what each measurement means, and the full list of limitations:
**[docs/RESULTS.md](docs/RESULTS.md)**.

## About this project

Shoe Atlas is a solo university project, written as coursework and kept as a portfolio piece for
internship and job applications. It is a working prototype, not a production system.

It is not affiliated with Zappos, with the University of Texas at Austin, or with any shoe retailer, and
no part of it is a commercial product or offered as a service.

## Data and licence

UT Zappos50K, from the University of Texas at Austin, used under its **academic, non-commercial** licence.
Dataset: <https://vision.cs.utexas.edu/projects/finegrained/utzap50k/>

**No licence is granted for this code.** It is published for reading, review and assessment; default
copyright applies and all rights are reserved. It comes with no warranty. If you want to use any of it for
something, ask.
