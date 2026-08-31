# Shoe Atlas

### 👟 [Try it live on Hugging Face](https://huggingface.co/spaces/AeonMuffinz/shoe-atlas)

Shoe Atlas looks at a product photo and works out eight things about the shoe: what kind it is, how high
the heel is, how it fastens, what it is made of, the shape of the toe, and a few more. It then uses those
predictions for two jobs. It checks a shop's own product data for mistakes, and it finds similar products
in the catalog.

Built on [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/), 50,025 shoe photos.
The interface is bilingual, Turkish by default.

**0.5745 mAP** on a held-out test set opened once &middot; **95.9%** top-1 on category and **0.0065**
calibration error on validation &middot; **107** attributes across 8 families &middot; **50,025** photos
indexed.

<sub>A personal project built to do during my internship. It was never handed over to the company and
stayed a project. Not a commercial product, and the dataset is for academic use only.</sub>

## Upload a photo

![Uploading a shoe photo, reading its attributes, then finding similar products](demo/upload-mode.gif)

Two things happen when you drop an image in.

**🏷️ It reads the attributes.** Eight groups of them, each with a confidence. Some are one of a kind,
like heel height, where a shoe has exactly one answer. Others can have several at once, like materials.
The model handles both, and the confidences shown are calibrated rather than raw model output.

**🔍 It finds similar products.** The photo is turned into a vector and matched against all 50,025
catalog images. Fine tuning on shoe attributes made this 2.6x better than the same network
untrained, and it beats CLIP.

## Check a catalog product

![Picking a catalog product, reading its attributes, then seeing which stored values agree and which conflict](demo/catalog-mode.gif)

Pick a product that is already in the catalog and you get the same two things, plus the one this project
was really built for.

**🏷️ The same attributes and 🔍 the same similar products**, read from the stored photo instead of an
upload.

**🚩 It checks the shop's own data.** Every online shop has products tagged wrong: the leather boot filed
as canvas, the three inch heel listed as flat. Here the model has something to compare against, so every
attribute is marked as **agreeing** with the catalog, **conflicting** with it, or **missing** and worth
filling in. That last one matters more than it sounds, because blank fields are as common as wrong ones.

Those flags are tested rather than trusted. Clean data is deliberately corrupted by a known amount, the
model is retrained on it, and the audit is scored on how much of the planted damage it finds. It
finds planted errors across all eight attribute families at every corruption rate tested.

## Run it yourself

You need [uv](https://docs.astral.sh/uv/) and two archives from the
[UT Zappos50K page](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/): `ut-zap50k-data.zip`
and `ut-zap50k-images-square.zip`. Extract both into `data/extracted/` so it looks like this:

```
data/extracted/
├── ut-zap50k-data/
│   ├── meta-data-bin.csv
│   └── meta-data.csv
└── ut-zap50k-images-square/
```

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
