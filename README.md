# Shoe Catalog Intelligence

Multi-label shoe attribute extraction, catalog audit, and same-product retrieval on UT Zappos50K.

Eight attribute families are predicted from a single product photograph. Three of them are strictly
exclusive and use a softmax head; the other five are genuinely multi-valued and use masked BCE. The
same model then drives two downstream products: an audit that flags catalog entries whose stored
attributes disagree with the image, and a retrieval index over the catalog.

> **Status.** The data pipeline, training, evaluation and the baselines are in place. The comparison
> grid is not complete, so this README does not yet carry a results table. Numbers quoted below are
> only those that have actually been measured, and each says what it was measured on.

## Quickstart

```bash
uv sync
```

```bash
uv run python -m src.prepare_data
```

```bash
uv run python -m src.train --config configs/convnext_tiny.yaml
```

```bash
uv run python -m src.evaluate --run artifacts/runs/convnext_tiny_s42
```

Baselines, none of which need a training config:

```bash
uv run python -m src.prevalence_floor
```

```bash
uv run python -m src.clip_zeroshot
```

```bash
uv run python -m src.linear_probe
```

Checks:

```bash
uv run ruff check .
```

```bash
uv run pytest
```

The test suite runs without the dataset. Tests that need the real download are marked `realdata` and
skipped by default.

## Evaluation protocol

Every run writes `artifacts/runs/{name}/evaluation.json` through one shared scoring function, so the
comparison table reads each run the same way. The contract is asserted rather than assumed: a run that
omits a key or invents an undeclared one fails at write time.

Selection maximises validation mAP over the 81 BCE labels, subject to validation mAP over the three
exclusive families staying within 1% of its running peak. On the reference run that meant taking epoch
7 at 0.4464 BCE mAP instead of epoch 11 at 0.4552. We deliberately gave up 0.0088 BCE mAP to protect
1.1% of exclusive-family mAP, because those three families feed the stronger of the two audit backends.

mAP is computed on uncalibrated probabilities. Calibration measurably moves average precision, by under
0.008 per label on the reference run but systematically: isotonic regression collapses distinct scores
into ties, and temperature scaling renormalises across a row while per-label AP ranks down a column, so
row-wise monotonicity does not imply column-wise monotonicity. `map_calibrated` is reported alongside
for transparency and is never the comparison number.

Macro F1 is reported twice. Thresholds and calibrators fitted on validation and scored on the same rows
come out optimistic, so both are refitted inside a 5-fold split of validation and scored on each
held-out fold. On the reference run that is 0.4827 fitted against 0.4589 out of fold, a gap of 0.0238.
The comparison table carries the out-of-fold figure.

The test set is opened once, for the winner only, and the lock is mechanical rather than a matter of
discipline: `--unlock-test` refuses unless `artifacts/winner.json` exists and names the run being
scored.

## Measured negatives

These cost time to establish and are recorded so they are not re-opened.

**Rebalancing the loss does not resolve the conflict between the two head groups.** Two probes at
`softmax_weight` 1.5 and 3.0, identical in every other respect, moved exclusive-family mAP at the
selected epoch from 0.8251 to 0.8262 to 0.8269, a gain of 0.0018 for three times the gradient weight,
while BCE mAP fell from 0.4464 to 0.4334 to 0.4160. The exchange rate is roughly 17 to 1 against us.
The hypothesis that the exclusive families decay because their gradient share shrinks does not survive
the numbers: they were already 52.5% of the training loss at the epoch their mAP peaks. The divergence
is structural, and the constrained selection rule is the mitigation rather than a stopgap.

**The directory structure yields no free audit signal.** The image path encodes Category and
SubCategory, and it agrees with the CSV on 50,025 of 50,025 rows. Zero disagreements. The path does
yield Brand, which appears nowhere in the CSV, and Brand is kept as a stratification variable only.

## Limitations

**The linear probe trains without augmentation.** Its features are extracted once from unaugmented
images and cached, while the fine-tuned runs are trained with random resized crops and horizontal
flips. This is a deliberate deviation from the otherwise-constant recipe. It disadvantages the probe,
so a competitive probe result is stronger than it looks rather than weaker. The head is 82,283
parameters, which is bit for bit the same head the fine-tuned run warms up before unfreezing, and is
small enough that augmentation's regularising role is close to irrelevant. The deviation is stated
here rather than left for a reader to infer from the code.

**CLIP zero-shot excludes one whole family.** Thirteen surviving Insole labels are not scoreable by
prompt: Poron, EVA, polyurethane, gel and memory foam are not distinguishable from one another in a
product photograph, even when the footbed is fully in frame, which on sandals and slippers it usually
is. The exclusion is family-level rather than a scattered mask, and the run reports mAP over all 107
labels and over the 94 scoreable ones separately, because a single number would let those thirteen
depress the baseline and be misread as a result about CLIP.

**CLIP's prompt template is chosen on validation.** That makes its headline mAP threshold-free but not
wholly untouched by validation. The bare no-template figure is reported alongside it so the value of
template selection is visible rather than absorbed into one number. CLIP's macro F1 is separately
labelled validation-calibrated, since its thresholds are swept on validation like every other run's.

**Two attribute pairs have a distinguishability ceiling, not a prompt-quality ceiling.** Moc Toe and
Algonquin differ only in whether the same U-shaped toe seam is puckered or a flat raised ridge.
Prewalker and Firstwalker differ in sole stiffness, which a photograph shows poorly. A low score on
these is a limit of the medium.

**Terry is confounded with slippers.** 91% of Terry positives are Slipper Flats, against slippers being
2.6% of the catalog, a 35x concentration. A correct Terry prediction may be recognising a slipper.

**44 of 151 labels were dropped** before training, at a threshold of 50 positives chosen from the
training split alone. Material lost 31 of its 63 labels, which is the single largest information loss
in the pipeline; anyone comparing our Material numbers against another system's needs to know the
vocabulary was halved first. The full dropped list is written to `artifacts/eda.json`.

**Retrieval ground truth measures same-product matching, not general visual similarity.** Relevance is
defined as sharing a ProductID, which is free and unambiguous but is not the same question as "does
this look similar". Colour is not one of the eight families, so no colour-sensitive evaluation is
possible from this metadata at all.

**The retrieval index covers the whole catalog, training split included.** That is the right behaviour
for a demo and the wrong thing to read as a held-out result.

**Runs are seed-controlled but not bitwise deterministic.** Full CUDA determinism costs real throughput
and buys nothing that seed-spread reporting does not already give. Seeds are fixed and recorded with
each run.

**One run is archived rather than deleted.** `artifacts/runs/convnext_tiny/` predates both the current
selection rule and the `{stem}_s{seed}` naming rule. It is marked `archived` in its summary and is
excluded from the comparison table rather than being reconciled by compatibility code.

## Data and licence

UT Zappos50K, from the University of Texas at Austin. The dataset is licensed for **academic,
non-commercial use**, and this project uses it on those terms. Download page:
`https://vision.cs.utexas.edu/projects/finegrained/utzap50k/`

Working set is the joined and deduplicated 50,025 rows: 50,066 image files against 50,025 metadata
rows, the gap being 35 duplicate CIDs from a URL-encoding artifact in one brand directory plus 6 images
with no metadata. Splits are 70/15/15 by `GroupShuffleSplit` on `product_id`, so no product appears in
two splits; a test asserts it.
