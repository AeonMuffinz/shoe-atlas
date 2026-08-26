# Shoe Atlas

Shoe Atlas looks at a product photo and works out eight things about the shoe: what kind it is, how high
the heel is, how it fastens, what it is made of, the shape of the toe, and a few more. It then uses those
predictions for two jobs. It checks a shop's own product data for mistakes, and it finds the same shoe in
other colours.

Built on [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/), 50,025 shoe photos.
The interface is bilingual, Turkish by default.

<!-- GIF 1: upload mode, drop a photo and see the attributes and similar shoes -->

<!-- GIF 2: catalog mode, pick a product and see the audit flag a wrong attribute -->

<sub>A personal project built to do during my internship. It was never handed over to the company and
stayed a project. Not a commercial product, and the dataset is for academic use only.</sub>

## What it does

**Reads attributes from a photo.** Eight groups of them. Some are one of a kind, like heel height, where a
shoe has exactly one answer. Others can have several at once, like materials. The model handles both, and
it says how confident it is about each.

**Finds mistakes in a shop's product data.** Every online shop has products tagged wrong: the leather boot
filed as canvas, the three inch heel listed as flat. The model compares what it sees against what the
catalog claims, and flags the disagreements.

Those flags are tested rather than trusted. Clean data is deliberately corrupted by a known amount, the
model is retrained on it, and the audit is scored on how much of the planted damage it finds.

**Finds the same shoe in other colours.** Search the catalog by image. Fine tuning on shoe attributes made
this 2.6x better than the same network untrained, and it beats CLIP.

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

Every run, how the model was selected, and what each measurement means:
**[docs/RESULTS.md](docs/RESULTS.md)**.
