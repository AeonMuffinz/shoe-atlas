# Shoe Atlas — detailed results

The full evaluation behind the summary in the main README: every run, how the winner was chosen, what
each downstream product measures, and what the project did not do.

Eight attribute families are predicted from a single product photograph. Three are strictly exclusive and
use a softmax head; the other five are genuinely multi-valued and use masked BCE. The same model drives
two downstream products: an audit that flags catalog entries whose stored attributes disagree with the
image, and a retrieval index over the catalog. A bilingual Gradio interface sits on top, Turkish by
default.

This is a solo university project. The goal was a working prototype with an honest evaluation, and the
honesty is the part worth reading: several results below are negatives, two headline numbers turned out
smaller than they first looked, and the limitations section is long on purpose. For a short overview,
read the main [README](../README.md) instead.

## Quickstart

```bash
uv sync
```

```bash
uv run python -m src.prepare_data
```

```bash
uv run python -m src.train --config configs/convnext_base_ar1.yaml
```

```bash
uv run python -m src.evaluate --run artifacts/runs/convnext_base_ar1_s42
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

Downstream products:

```bash
uv run python -m src.audit --run artifacts/runs/convnext_base_ar1_uniform10_s42 --corrupt-rate 0.10 --corrupt-seed 4242 --corrupt-type uniform
```

Retrieval is two steps: `src.retrieval` extracts and persists the embeddings for all three sources,
`scripts.retrieval_score` ranks and scores them.

```bash
uv run python -m src.retrieval --run artifacts/runs/convnext_base_ar1_s42
```

```bash
uv run python -m scripts.retrieval_score --run artifacts/runs/convnext_base_ar1_s42
```

```bash
uv run python -m src.compare_runs
```

```bash
uv run python -m src.app
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

## Results

Validation split, 7,658 rows, 107 surviving labels. Every figure comes from that run's `evaluation.json`,
written through one shared scoring function whose key set is asserted at write time.

| run | mAP (107) | mAP softmax (26) | mAP bce (81) | macro F1 oof | ECE oof | top-1 Category | top-1 SubCategory | top-1 HeelHeight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prevalence floor, no model | 0.0927 | 0.1154 | 0.0854 | 0.1000 | 0.0044 | 0.6119 | 0.2458 | 0.3238 |
| CLIP zero-shot | 0.3126 | 0.6443 | 0.2061 | 0.2246 * | 0.1409 | 0.9000 | 0.6994 | 0.4415 |
| linear probe, frozen trunk | 0.4827 | 0.7811 | 0.3869 | 0.4100 | 0.0094 | 0.9499 | 0.8389 | 0.6937 |
| ConvNeXt-Tiny | 0.5385 | 0.8251 | 0.4464 | 0.4589 | 0.0083 | 0.9557 | **0.8612** | 0.7592 |
| Swin-Tiny | 0.5319 | 0.8293 | 0.4365 | 0.4479 | 0.0096 | 0.9547 | 0.8518 | 0.7588 |
| ConvNeXt-Tiny, lr 1e-4 | 0.5284 | 0.8238 | 0.4336 | 0.4471 | 0.0085 | 0.9536 | 0.8537 | 0.7584 |
| ConvNeXt-Tiny, input 160 | 0.5327 | 0.8227 | 0.4396 | 0.4497 | 0.0080 | 0.9566 | 0.8558 | 0.7568 |
| ConvNeXt-Tiny, layer_decay 0.6 | 0.5223 | 0.8175 | 0.4276 | 0.4492 | 0.0088 | 0.9555 | 0.8571 | 0.7482 |
| ConvNeXt-Tiny, no aspect jitter | 0.5402 | 0.8310 | 0.4468 | 0.4553 | 0.0089 | 0.9553 | 0.8590 | 0.7627 |
| ConvNeXt-Base | 0.5633 | 0.8244 | 0.4795 | 0.4867 | 0.0069 | 0.9580 | 0.8558 | 0.7506 |
| ConvNeXt-Base, pos_weight cap 50 | 0.5607 | 0.8161 | 0.4787 | 0.4830 | 0.0069 | 0.9559 | 0.8549 | 0.7525 |
| ConvNeXt-V2-Base @192 | 0.5349 | **0.8347** | 0.4386 | 0.4456 | 0.0103 | 0.9559 | 0.8568 | 0.7588 |
| ConvNeXt-Large @192, batch 32 | 0.5369 | 0.8257 | 0.4443 | 0.4578 | 0.0105 | 0.9536 | 0.8557 | 0.7480 |
| **ConvNeXt-Base, no aspect jitter** *(winner)* | **0.5659** | 0.8228 | **0.4834** | **0.4920** | **0.0065** | **0.9586** | 0.8565 | **0.7683** |

Every figure above is on **validation**, which is the split selection was made from. Only the winner has a
test score — 0.5745 mAP, tabulated in full below — and every other row reads *not evaluated, test opened
once for the winner* in `artifacts/comparison.md` rather than being left blank.

\* CLIP's macro F1 is validation-calibrated, not zero-shot: its thresholds are swept on validation like
every other run's. Its mAP is threshold-free and genuinely zero-shot, but the prompt template was chosen
on validation, so the bare no-template figure of 0.2991 is reported beside the selected 0.3126.

The four largest backbones are **capacity probes, not grid rows.** The grid holds every candidate at
similar capacity so architecture is not confounded with parameter count; ConvNeXt-Base is 3.1x the
reference and ConvNeXt-V2 changes the pretraining recipe, so those break the premise by design and are
reported under their own heading in `artifacts/comparison.md`.

Two runs carry no evaluation at all by design — the `softmax_weight` 1.5 and 3.0 probes were scored on
their training curves only. One run, `artifacts/runs/convnext_tiny/`, is marked `archived` because it was
selected on `val_loss` before the current rule existed; it is excluded rather than reconciled.

### The winner, and what "winner" does and does not mean here

**`convnext_base_ar1_s42` leads on validation mAP at 0.5659.** It leads on aggregate mAP, `map_bce` at the
selected epoch, macro F1 both fitted and out of fold, calibration error on all three bases, top-1 Category
and top-1 HeelHeight.

**It is the leader on the stated metric, not the output of the selection rule, and that distinction is
real.** The pre-registered tie-break is *if the mAP gap sits inside the seed standard deviation, pick the
smaller and faster model*, and it requires three seeds per finalist. The margin over `convnext_base_s42`
is **0.0026 of mAP** and **the seed sweep was cancelled**, so seed variance in this project is unmeasured
and will stay unmeasured. Whether 0.0026 is a real ordering or noise is unanswerable. The run is preferred
on a **design** ground rather than a measured one: it trained under the augmentation this project
established as correct. It must not be described as "selected", and no seed spread may be quoted, because
none exists.

It also does not lead everywhere, and the places it loses are worth stating:

- **`map_softmax` 0.8228 — second lowest of the twelve fine-tuned runs.** The winner is the weakest strong
  run on the three exclusive families.
- **Peak `map_bce` 0.4898, second** behind ConvNeXt-V2-Base's 0.4918. Peak is reported and never scored:
  it occurs at an epoch the guard rejects so no procedure could select it, a maximum over a noisy curve is
  upward-biased in the number of epochs, and it became interesting only after it reordered a table.
- **top-1 SubCategory 0.8565** against ConvNeXt-Tiny's 0.8612.

### The test set, opened once

The test set was untouched for the entire project and was opened **once, on 26 August 2026, for the winner
only**. The lock is mechanical rather than a matter of discipline: `--unlock-test` refuses unless
`artifacts/winner.json` exists and names the run being scored, and it was verified to refuse
`convnext_base_s42` and `convnext_tiny_s42` before it was used. No losing run has a test score and none
will. "Once" meant once in total across subsystems, so that single pass covered the winner's test
evaluation **and** the test-split audit below.

| | validation | **test** | delta |
|---|---:|---:|---:|
| rows | 7,658 | 7,483 | |
| **mAP (107)** | 0.5659 | **0.5745** | +0.0086 |
| mAP softmax (26) | 0.8228 | **0.8453** | +0.0225 |
| mAP bce (81) | 0.4834 | **0.4875** | +0.0041 |
| macro F1 out-of-fold | 0.4920 | **0.4965** | +0.0045 |
| calibration error oof | 0.0065 | 0.0071 | +0.0006 |
| top-1 Category | 0.9586 | 0.9562 | −0.0024 |
| top-1 SubCategory | 0.8565 | 0.8562 | −0.0003 |
| top-1 HeelHeight | 0.7683 | 0.7642 | −0.0041 |

**The model generalises.** Test is slightly *better* than validation on every ranking metric and slightly
worse on the three top-1 accuracies, with nothing that resembles a collapse. The largest single move is
the exclusive families at +0.0225, which is the group the constrained selection rule spent 0.0313 of BCE
mAP to protect.

**Two cautions against over-reading this.** The differences are small and there is **no seed band around
any of them**, so "test is better than validation" is one draw of one model and not a claim that the gap
is real. And a single split of 7,483 rows carries its own sampling noise — HeelHeight top-1 is measured
over 4,000-odd observed rows, where a binomial standard error near p = 0.76 is already about 0.007, larger
than the −0.0041 shown.

**Nothing was tuned after these numbers were seen**, and nothing can be: the seed sweep is cancelled, no
further backbone run is commissioned, and the winner was declared on validation before the lock was armed.
If any model here had been changed after this table existed, it would no longer be a test set, and this
sentence would say so.

## Headline numbers

**What fine-tuning bought: 0.0557 mAP.** 0.5385 fine-tuned against 0.4827 for a linear probe on frozen
ImageNet features, same backbone, same objective, same split, same seed, same selection rule. That is 10.3%
relative and roughly 56x the per-label noise floor. Frozen features are not enough.

**The gain is uneven, and the aggregate misrepresents both halves.** Share of the floor-to-fine-tuned range
that fine-tuning adds on top of a frozen trunk: **6.2% on the 26 exclusive labels, 16.5% on the 81 BCE
labels.** The blended 12.5% describes neither regime. This is why every mAP in this project is reported per
head group and never as one scalar.

**What the selection rule cost, at its true size.** Selection maximises validation mAP over the 81 BCE
labels subject to mAP over the exclusive families staying within 1% of its running peak. On the reference
run that took epoch 7 at 0.4464 `map_bce` over the BCE optimum at epoch 26, which reaches 0.4777:

| | epoch 7, selected | epoch 26, BCE optimum | given up |
|---|---:|---:|---:|
| `map_bce` | 0.4464 | 0.4777 | **0.0313** |
| mAP (107) | 0.5385 | 0.5592 | **0.0207** |
| `map_softmax` | 0.8251 | 0.8130 | protected **0.0121** |

**That is roughly 2.6 to 1 against the aggregate**, paid deliberately because the exclusive families feed
cleanlab's multiclass path, the stronger of the two audit backends. A figure of 0.0088 appears in earlier
drafts of this project and **understates the trade** — it compares against epoch 11 of a run that early
stopping truncated. 0.0313 is the honest number.

**Rebalancing the loss does not fix the head-group divergence: a 17-to-1 exchange rate against us.** Two
probes at `softmax_weight` 1.5 and 3.0 moved exclusive-family mAP from 0.8251 to 0.8262 to 0.8269 — a gain
of **0.0018** — while BCE mAP fell 0.4464 to 0.4334 to 0.4160, a loss of **0.0304**. This is a measurement
and it stands on its own. An earlier explanation attributing it to trunk drift was **withdrawn**: direct
gradient measurement found no negative cosine in any block on either architecture, so there is no
measurable conflict for such a mechanism to act through. The current reading is differential overfitting —
26 labels on an easier, mutually exclusive task sitting near their ceiling overfit sooner than 81 sparse
ones still learning — and it is offered as a hypothesis with one pre-registered result in its favour.

**Capacity is a much smaller lever than it first appeared.** ConvNeXt-Base over ConvNeXt-Tiny reads
**+0.0331 `map_bce` at the selected epoch**, more than twice the pre-registered +0.015 band. Measured
aspect-correct at the unconstrained peak it is **+0.0047**, below that band. The difference is where the
guard cuts each curve, not what the parameters buy: the two runs' guard costs are 0.0383 and 0.0064, and
their difference of 0.0319 accounts for the gap. **Both numbers belong in the record** — +0.0331 is what a
practitioner operating this pipeline actually gets, +0.0047 is what capacity is worth to the underlying
model — and quoting only the first overstates it.

**The one intervention that moved HeelHeight was removing an augmentation nobody chose.**
`RandomResizedCrop`'s `ratio` argument was absent from every config, so fifteen runs trained under
torchvision's default aspect jitter. Measured over 20,000 draws from the production transform it spans
**0.91 of a HeelHeight bracket width** at a 2-inch heel — and 78.3% of HeelHeight errors were exactly one
bracket out. Removing it moved HeelHeight top-1 **+0.0177** against a band of +0.0112 derived before the
run, and **the entire gain came out of the one-step error bucket**: 87 one-step errors disappeared while
two-step rose by 2 and three-or-more by 3, a factor of thirty apart. That error *shape* was pre-registered
and is stronger evidence than the primary. **It did not replicate at ConvNeXt-Tiny** (+0.0035, a null), so
the claim is that it moved HeelHeight at base capacity — not that the augmentation was wrong for every run.

## Catalog audit

The headline deliverable. A raw disagreement list is not evidence, because most disagreements are model
error and nothing about reading the list tells you which is which. So ground truth is manufactured: corrupt
a known fraction of labels, train on the corrupted labels, run the pipeline blind, and score detection
against the known corruption.

Six runs, twelve epochs each at a fixed epoch so training duration is held constant, at three corruption
rates and two corruption schemes. **Detection recall, per family, never pooled:**

| family | backend | unif 5% | unif 10% | unif 20% | conf 5% | conf 10% | conf 20% |
|---|---|---:|---:|---:|---:|---:|---:|
| Category | multiclass | 0.895 | 0.890 | 0.848 | 0.909 | 0.890 | 0.837 |
| SubCategory | multiclass | 0.934 | 0.926 | 0.880 | 0.831 | 0.858 | 0.827 |
| HeelHeight | multiclass | 0.840 | 0.843 | 0.852 | 0.807 | 0.799 | 0.774 |
| Gender | multilabel | 0.840 | 0.833 | 0.787 | 0.825 | 0.840 | 0.777 |
| Closure | multilabel | 0.824 | 0.767 | 0.742 | 0.804 | 0.793 | 0.681 |
| ToeStyle | multilabel | 0.833 | 0.758 | 0.662 | 0.810 | 0.729 | 0.653 |
| Material | multilabel | 0.774 | 0.673 | 0.593 | 0.726 | 0.676 | 0.577 |
| Insole | multilabel | 0.694 | 0.656 | 0.543 | 0.657 | 0.658 | 0.569 |

**The audit detects planted noise on all eight families, at every rate, under both schemes.**

**Detection numbers are reported per family group and never pooled, and the reason is not presentational.**
The three exclusive families go to cleanlab's multiclass path, which estimates a full joint distribution
and is its best-tested code. The five multi-valued families go to the multilabel path, which decomposes to
one-against-all and has weaker joint structure. **The two also count different things** — multiclass
detects per row, multilabel per cell — so the blocks above are not comparable as ratios, only as trends. A
single headline number would average two backends of different strength and flatter the weaker one.

Three further results, each of which changes how the table should be read:

- **The gap between the two backends roughly doubles as noise rises**, from +0.084 at 5% to +0.161 at 20%
  under confusion weighting. The weaker backend degrades three to four times faster.
- **Precision and precision@100 rise with the corruption rate and this is not the audit improving.** At 20%
  there is four times as much to find, so a flag is four times more likely to be right by construction.
  Precision is not comparable across rates.
- **Most of the false-alarm rate is model error, not corruption.** At 5% corruption, between 50% and 105%
  of each family's false-alarm rate is already present with **zero** planted noise. On HeelHeight it is
  86.5%; on Insole the corrupted runs flag *fewer* clean cells than the clean model does. This is the
  project's own founding claim — that a raw disagreement list is mostly your model's mistakes — measured
  directly for the first time.

**Confusion-weighted corruption is harder to detect than uniform, as a direction and not robustly family by
family.** The sign is correct in 8 of 9 comparisons against a noise band measured from the design itself,
but no family clears that band at all three rates. Quoting a single rate would overstate it either way.

### The false-alarm floor replicates on the test split

The zero-corruption floor was recomputed on the test split inside the single unlock, so the figure above is
not resting on one split. False-alarm rate with nothing planted:

| family | validation | test | delta |
|---|---:|---:|---:|
| Category | 0.0150 | 0.0150 | −0.0000 |
| SubCategory | 0.0815 | 0.0787 | −0.0028 |
| HeelHeight | 0.0996 | 0.0957 | −0.0040 |
| Gender | 0.0329 | 0.0371 | +0.0042 |
| Insole | 0.0350 | 0.0361 | +0.0011 |
| Closure | 0.0078 | 0.0096 | +0.0018 |
| Material | 0.0189 | 0.0184 | −0.0005 |
| ToeStyle | 0.0162 | 0.0165 | +0.0003 |

**All eight families reproduce within ±0.0042**, so how often the audit cries wolf on clean data is a
property of the model rather than of the split it was measured on.

**What this is not.** It is a false-alarm floor, not a detection result. The winner trained on clean labels,
so there is no planted noise in it to find, and **no detection recall is reported on test at all.**

**The Phase A detection curves are validation-only, and that is a limitation rather than an oversight.**
Auditing detection on test would need the six corrupted-label runs each scored on test, and the lock names
exactly one run by design — widening it to six non-winner runs would change the project's central safety
mechanism for a replication rather than a new measurement, since validation is already fully out-of-sample
for those models. Reported as not attempted.

**Phase B was not attempted.** No K-fold, so no product outside validation has an out-of-sample prediction.

## Retrieval

Same-product retrieval over the whole catalog, three embedding sources so that fine-tuning is separated
from pretraining corpus and from architecture. Query set is the 38,655 images belonging to the 13,153
products with two or more colorways; the 11,370 single-colorway products are excluded because they have no
relevant item to find. mAP@10, macro over queries, gallery is all 50,025 images.

| source | mAP@10 | mAP@10 over products | R@10 capped | R@10 uncapped | CMC@1 | CMC@10 |
|---|---:|---:|---:|---:|---:|---:|
| **fine-tuned** | **0.6620** | 0.6935 | 0.8250 | 0.8175 | 0.6864 | 0.9354 |
| CLIP ViT-B/32 | 0.3564 | 0.3735 | 0.5635 | 0.5590 | 0.3806 | 0.7519 |
| ImageNet-pretrained | 0.2568 | 0.2721 | 0.3863 | 0.3825 | 0.3121 | 0.5733 |

Paired bootstrap over the 38,655 queries, 2,000 resamples, 95% interval:

| | comparison | Δ mAP@10 | 95% CI | reading |
|---|---|---:|---|---|
| **C1** | fine-tuned − pretrained | **+0.4052** | [+0.4017, +0.4089] | the clean comparison; fine-tuning wins |
| **C2** | fine-tuned − CLIP | +0.3057 | [+0.3022, +0.3093] | significant and **uninterpretable alone** |
| **C3** | pretrained − CLIP | −0.0996 | [−0.1025, −0.0964] | CLIP wins without any fine-tuning |

C1 is the only clean one: same architecture, same corpus, same pipeline, fine-tuning the only difference.
C2 moves fine-tuning, corpus and architecture at once and is reported without attribution. C3 exists to
make C2 readable at all.

**A registered prediction was inverted here and the reason is worth stating.** The project expected
fine-tuning to *lose*, on the reasoning that training on attributes pulls the space toward attribute
clusters and away from instance identity. It did not lose. The ground truth is why: a relevant item is the
same ProductID at a different ColorID — the same shoe in another colour — and two colorways share **every
one of the eight families**. They differ only in colour, and colour is not one of the families. So an
encoder trained on those families is trained to map colorway siblings to the same point. **Attribute
clustering and same-product clustering are the same clustering here.** The warned-of mechanism is real; it
simply points the other way for this ground truth, and **this does not generalise to visual similarity
retrieval.**

**About a fifth of C1 is memorisation.** The gallery includes training images, so most queries are images
the fine-tuned backbone has seen. Split by provenance — and products never span splits, so a query's
targets are always in its own split:

| source | pooled | train | validation | test |
|---|---:|---:|---:|---:|
| fine-tuned | 0.6620 | 0.6818 | 0.6126 | 0.6213 |
| CLIP | 0.3564 | 0.3546 | 0.3603 | 0.3608 |
| ImageNet-pretrained | 0.2568 | 0.2537 | 0.2674 | 0.2601 |

C1 is **+0.4281 on seen queries and +0.3452 / +0.3612 on unseen ones**. A control makes that estimate
conservative rather than generous: if the unseen strata were simply harder, every source would drop on
them, and the two that never trained on anything both score *slightly better* there. Only the fine-tuned
source falls. **Both numbers are legitimate and answer different questions** — the pooled figure is what
the deployed demo does on this catalog, where most queries genuinely are indexed training images, and the
unseen figure is what the representation is worth on products it has never seen. Use the unseen one when
the question is representation quality.

## Interface

`uv run python -m src.app`. Turkish by default with an explicit language switch that relabels the entire
interface, not just the outputs. Two modes, because an audit needs a stored value to contradict:

- **Upload mode** takes any photo and shows predicted attributes with calibrated confidences plus the
  nearest catalog products. **No audit warning**, because there is nothing to compare against and
  comparing against a retrieved product's labels would report retrieval errors as catalog errors.
- **Catalog mode** additionally compares the stored row against the prediction, marking each family as
  agreeing, contradicting, or blank-with-a-suggestion.

Confidences shown are calibrated probabilities, never raw sigmoid. The head design is respected in the
display: the three exclusive families always show exactly one value chosen by argmax within the family,
and the five multi-valued families show whatever clears each label's own threshold — which may be several
values or none, and "none" is displayed as such rather than padded with a low-confidence guess.

Attribute names come through a domain glossary of 115 terms — 107 labels and 8 family headings — each with
Turkish and English display names. **Every Turkish term was supplied by a domain speaker rather than
composed**, with its provenance recorded: the retail usage or named retailer's filter value it came from,
or an explicit note that no settled Turkish equivalent exists. Neither the worksheet script nor the
importer will generate a term. Heel heights are shown in centimetres in Turkish because Turkish footwear
retail sizes heels that way — a unit change rather than a translation, and the underlying label is
unchanged.

## Limitations

**Runs are single-seed and seed variance was never measured.** This is the largest limitation and it is
not the same as the determinism note below. The seed sweep was cancelled, so no run in this project has a
band around it. The consequence is concrete: the winner leads by 0.0026 mAP and **the pre-registered
tie-break that exists to decide exactly that margin cannot be applied, permanently.** Every intervention
verdict rests on one seed per configuration, with effect bands derived from spread across *different*
configurations rather than repeated runs of the same one. The two largest effects — fine-tuning at +0.0557
and retrieval's C1 at +0.4052 — are far outside any plausible seed band, but the small margins are not.

**Runs are seed-controlled but not bitwise deterministic.** Full CUDA determinism costs real throughput
and buys nothing that seed-spread reporting would have given. Seeds are fixed and recorded per run.
Evaluation *is* reproducible: re-scoring a fixed checkpoint returns byte-identical probabilities.

**Per-label AP carries a numerical noise floor of about 0.001.** Evaluation runs under bfloat16 autocast,
which quantises the scores mAP ranks on — a median of 846 distinct values per BCE column against 7,657 in
float32. The aggregate is unaffected (float32 moves it by 0.00003) but individual labels move by 0.001 on
average and up to 0.0097. No per-label difference below 0.001 is a finding.

**44 of 151 labels were dropped** before training, at a threshold of 50 training positives chosen from the
training split alone. By family: SubCategory lost 6 of 21, Closure 5 of 19, Insole 1 of 14, ToeStyle 1 of
19, and **Material lost 31 of its 63** — the single largest information loss in the pipeline. Anyone
comparing these Material numbers against another system's needs to know the vocabulary was halved first.
Category, HeelHeight and Gender lost nothing. The full list is in `artifacts/eda.json`.

**22 of the 81 BCE labels are not really calibrated on validation, and 21 on test.** No label lacked
positives on either split, so none fell back to a default threshold — but labels with fewer than 50
positives in the split being fitted receive **identity calibration** rather than a fitted isotonic, because
fitting on that few would fit noise. Material supplies 13 of the validation set. **The set is a property of
the split, not of the model**, and the two do not coincide: test drops `Material.Hair.Calf` and
`Material.Microfiber` and gains `Material.Nappa`. Both sets are named in `evaluation.json` under
`identity_calibrated_labels`, per split.

**CLIP zero-shot excludes one whole family.** Thirteen surviving Insole labels are not scoreable by prompt:
Poron, EVA, gel and memory foam are not distinguishable from one another in a photograph even when the
footbed is fully in frame, which on sandals and slippers it usually is. mAP is reported over all 107 labels
and over the 94 scoreable ones separately.

**CLIP's probabilities are unusable as confidences and never feed anything downstream.** Its calibration
error is 0.1194 fitted and 0.1409 out of fold against 0.0044–0.0094 for the trained models — 14 to 27 times
worse. Its ranking metrics are fine; its probabilities are not, and a better CLIP mAP would not change that.

**Two attribute pairs have a distinguishability ceiling rather than a prompt-quality one.** Moc Toe and
Algonquin differ only in whether the same U-shaped seam is puckered or a flat ridge. Prewalker and
Firstwalker differ in sole stiffness, which a photograph shows poorly.

**Terry is confounded with slippers.** 91% of Terry positives are Slipper Flats against slippers being 2.6%
of the catalog, a 35x concentration, so a correct Terry prediction may be recognising a slipper.

**HeelHeight's residual is mostly the label scheme, not recognition.** 78.3% of its errors are one physical
bracket away and only 5.2% of rows are wrong by more than one. The brackets are hand-decoded ranges like
`1in...1.3.4in`, so adjacent brackets are separated by a boundary someone chose. Architecture, resolution
and capacity were all tried on this family and none moved it; only removing the aspect jitter did, and that
recovered 87 of 949 one-step errors — 9.2%. The label-semantics reading survives as the majority
explanation but is no longer the whole one.

**Brand stratification is inconclusive.** 594 brands appear in validation and the largest has 181 rows, so
none reaches a 200-row floor. At 30 rows a single error moves accuracy by 0.033, so most of the apparent
spread is sampling noise. Reported as required and as inconclusive; a brand effect could only be
established on the full catalog.

**The linear probe trains without augmentation.** Its features are cached once from unaugmented images
while every fine-tuned run trains with random resized crops and flips. This is a deliberate deviation and
it **disadvantages the probe**, so the 0.0557 fine-tuning gap is a conservative estimate rather than an
inflated one.

**Retrieval ground truth measures same-product matching, not general visual similarity.** Relevance is
sharing a ProductID, which is free and unambiguous but is not the question "does this look similar".
**Colour is not one of the eight families, so no colour-sensitive evaluation is possible from this metadata
at all** — and that is also why the fine-tuned encoder wins C1, so the win and the limitation are the same
fact seen twice.

**The retrieval index covers the whole catalog, training images included.** Right for a demo, wrong to read
as a held-out result. The provenance table above is there so the distinction is visible rather than buried.

**The demo's catalog picker offers the first 400 products**, not all 50,025, because a dropdown that size is
unusable. The index itself is not limited — retrieval searches the whole catalog. The neighbour strip shows
five rather than ten, which is a display choice and not a change to how retrieval is scored.

**The selection rule degenerates on some runs.** It maximises BCE mAP subject to a guard, but on two runs
the feasible set collapsed to one and two admissible epochs, at which point the constrained objective is
not optimising anything — it returns the only admissible point. Those runs should be read as "the guard
chose this epoch". Fixing it needs a tolerance of 0.02, and by then the leader on the primary metric has
already changed twice, so **fixing the degeneracy and preserving the ordering are not simultaneously
available.** The tolerance was fixed at 0.01 by prior commitment and no run was re-selected.

**One run is archived rather than deleted.** `artifacts/runs/convnext_tiny/` predates both the current
selection rule and the naming rule; it is marked `archived` and excluded from the comparison table rather
than reconciled by compatibility code.

### Not attempted

Reported as not attempted rather than left to inference: Phase B K-fold cross-validation for the audit;
the seed sweep; `convnext_small` as the clean capacity probe; a fine-tuned CLIP image encoder; focal loss
and Asymmetric Loss as imbalance alternatives; partial BCE against masked BCE; probability-weighted
corruption for the five BCE families; adjacent-aware label smoothing on HeelHeight; the split-tower
architecture that the gradient diagnostic pointed at; and an out-of-distribution check against Amazon
Berkeley Objects.

## Data and licence

UT Zappos50K, from the University of Texas at Austin. The dataset is licensed for **academic,
non-commercial use**, and this project uses it on those terms. Download page:
`https://vision.cs.utexas.edu/projects/finegrained/utzap50k/`

Working set is the joined and deduplicated **50,025 rows**: 50,066 image files against 50,025 metadata
rows, the gap being 35 duplicate CIDs from a URL-encoding artifact in one brand directory plus 6 images
with no metadata row. Splits are 70/15/15 by `GroupShuffleSplit` on `product_id`, so no product appears in
two splits and a test asserts it. Training uses the square images as distributed, without cropping back to
native, because a non-square resize stretches the shoe vertically and that distortion lands directly on the
two shape-dependent families.
