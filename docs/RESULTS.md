# Shoe Atlas: results and method

The evaluation behind the summary in the main [README](../README.md): every run, how the model was
chosen, and what each measurement means.

Eight attribute families are predicted from one product photograph. Three are strictly exclusive and use
a softmax head; the other five are genuinely multi-valued and use masked BCE. The same model then drives
two downstream products, a catalog audit and a retrieval index.

## Results

Validation split, 7,658 rows, 107 surviving labels. Every figure comes from that run's `evaluation.json`,
written through one shared scoring function whose key set is asserted at write time.

| run | mAP (107) | mAP softmax (26) | mAP bce (81) | macro F1 oof | ECE oof | top-1 Cat | top-1 SubCat | top-1 Heel |
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

\* CLIP's macro F1 is validation-calibrated rather than zero-shot: its thresholds are swept on validation
like every other run's. Its mAP is threshold-free and genuinely zero-shot, though the prompt template was
chosen on validation, so the bare no-template figure of 0.2991 sits beside the selected 0.3126.

The four largest backbones are **capacity probes, not grid rows.** The grid holds every candidate at
similar capacity so architecture is not confounded with parameter count. ConvNeXt-Base is 3.1x the
reference and ConvNeXt-V2 changes the pretraining recipe, so both break that premise by design.

## The winner, and what that word does not mean here

`convnext_base_ar1_s42` leads on aggregate mAP, `map_bce`, macro F1, calibration on all three bases,
top-1 Category and top-1 HeelHeight.

**It is the leader on the stated metric, not the output of the selection rule.** The pre-registered
tie-break is *if the mAP gap sits inside the seed standard deviation, pick the smaller and faster model*,
and it needs three seeds per finalist. The margin over ConvNeXt-Base is **0.0026 of mAP** and the seed
sweep was cancelled, so seed variance is unmeasured and whether 0.0026 is a real ordering is unanswerable.
The run is preferred on a **design** ground rather than a measured one: it trained without the aspect
jitter that no config had ever recorded.

It also loses on three axes worth stating: `map_softmax` 0.8228 is the second lowest of the twelve
fine-tuned runs, peak `map_bce` is second behind ConvNeXt-V2-Base, and top-1 SubCategory is below
ConvNeXt-Tiny's.

## The test set, opened once

The test set was untouched for the whole project and opened once, for the winner only. The lock is
mechanical rather than a matter of discipline: `--unlock-test` refuses unless `artifacts/winner.json`
exists and names the run being scored, and it was verified to refuse losing runs before it was used.

| | validation | test | delta |
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

**The model generalises.** Test is slightly better than validation on every ranking metric and slightly
worse on the three top-1 accuracies, with nothing resembling a collapse.

Two cautions against over-reading it. There is **no seed band around any of these deltas**, so this is one
draw of one model. And a single split of 7,483 rows carries its own sampling noise: HeelHeight top-1 is
measured over roughly 4,000 observed rows, where the binomial standard error near p = 0.76 is about 0.007,
larger than the −0.0041 shown. All three top-1 deltas sit inside their own sampling error.

Nothing was tuned after these numbers were seen, and nothing can be: the seed sweep is cancelled and no
further backbone run is commissioned.

## What the numbers mean

**Fine-tuning bought 0.0557 mAP.** 0.5385 fine-tuned against 0.4827 for a linear probe on frozen ImageNet
features, with the same backbone, objective, split, seed and selection rule. That is 10.3% relative.
The probe is additionally handicapped, since its features are cached from unaugmented images, so the gap
is a conservative estimate rather than an inflated one.

**The gain is uneven, and the aggregate misrepresents both halves.** Share of the floor-to-fine-tuned
range that fine-tuning adds on top of a frozen trunk: **6.2% on the 26 exclusive labels, 16.5% on the 81
BCE labels.** The blended 12.5% describes neither regime, which is why every mAP here is reported per head
group and never as one scalar.

**The selection rule cost 0.0313 `map_bce`.** Selection maximises validation mAP over the 81 BCE labels
subject to mAP over the exclusive families staying within 1% of its running peak. On the reference run
that took epoch 7 at 0.4464 over the BCE optimum at epoch 26, which reaches 0.4777:

| | epoch 7, selected | epoch 26, BCE optimum | given up |
|---|---:|---:|---:|
| `map_bce` | 0.4464 | 0.4777 | **0.0313** |
| mAP (107) | 0.5385 | 0.5592 | **0.0207** |
| `map_softmax` | 0.8251 | 0.8130 | protected **0.0121** |

Roughly **2.6 to 1 against the aggregate**, paid deliberately because the exclusive families feed the
stronger of the two audit backends.

**Rebalancing the loss does not fix the split between the two head groups: a 17-to-1 exchange rate against
us.** Probes at `softmax_weight` 1.5 and 3.0 moved exclusive-family mAP 0.8251 → 0.8262 → 0.8269, a gain
of **0.0018**, while BCE mAP fell 0.4464 → 0.4334 → 0.4160, a loss of **0.0304**. Direct gradient
measurement then found no negative cosine in any block on either architecture, so there is no measurable
conflict for such a mechanism to act through. The current reading is differential overfitting, offered as
a hypothesis rather than a result.

**Capacity is a much smaller lever than it first looked.** ConvNeXt-Base over ConvNeXt-Tiny reads
**+0.0331 `map_bce` at the selected epoch**, more than twice the pre-registered band. Measured
aspect-correct at the unconstrained peak it is **+0.0047**, below that band. The difference is where the
selection guard cuts each curve, not what the parameters buy: the two guard costs are 0.0383 and 0.0064,
and their difference of 0.0319 accounts for the gap. Both belong in the record. The first is what a
practitioner operating this pipeline actually gets; the second is what capacity is worth to the model.

**The one intervention that moved HeelHeight was removing an augmentation nobody chose.**
`RandomResizedCrop`'s `ratio` argument was absent from every config, so fifteen runs trained under
torchvision's default aspect jitter. Measured over 20,000 draws it spans **0.91 of a HeelHeight bracket
width** at a two inch heel, and 78.3% of HeelHeight errors were exactly one bracket out. Removing it moved
HeelHeight top-1 **+0.0177** against a band of +0.0112 fixed before the run, and **the entire gain came out
of the one-step error bucket**: 87 one-step errors disappeared while two-step rose by 2 and three-or-more
by 3, a factor of thirty apart. That error shape was pre-registered and is stronger evidence than the
headline. It did **not** replicate at ConvNeXt-Tiny (+0.0035, a null), so the claim is that it moved
HeelHeight at base capacity, not that the augmentation was wrong everywhere.

## Catalog audit

The headline deliverable. A raw disagreement list is not evidence, because most disagreements are model
error and nothing about reading the list tells you which is which. So ground truth is manufactured:
corrupt a known fraction of labels, train on the corrupted labels, run the pipeline blind, and score
detection against the known corruption.

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

**Detection numbers are reported per family group and never pooled**, and the reason is not
presentational. The three exclusive families go to cleanlab's multiclass path, which estimates a full
joint distribution. The five multi-valued ones go to the multilabel path, which decomposes to
one-against-all and has weaker joint structure. The two also count different things, multiclass per row
and multilabel per cell, so the blocks are comparable as trends and not as ratios.

Three further readings, each of which changes how the table should be read:

- **The gap between the backends roughly doubles as noise rises**, from +0.084 at 5% to +0.161 at 20%
  under confusion weighting. The weaker backend degrades three to four times faster.
- **Precision rises with the corruption rate, and that is not the audit improving.** At 20% there is four
  times as much to find, so a flag is four times more likely to be right by construction. Precision is not
  comparable across rates.
- **Most of the false-alarm rate is model error, not corruption.** At 5% corruption, between 50% and 105%
  of each family's false-alarm rate is already present with **zero** planted noise. On HeelHeight it is
  86.5%; on Insole the corrupted runs flag *fewer* clean cells than the clean model does. This measures
  the project's own founding claim directly: a raw disagreement list is mostly your model's mistakes.

**Confusion-weighted corruption is harder to detect than uniform, as a direction and not robustly family
by family.** The sign is correct in 8 of 9 comparisons against a noise band measured from the design
itself, but no family clears that band at all three rates.

That zero-corruption floor was also recomputed on the test split and **reproduces within ±0.0042 on all
eight families**, so how often the audit cries wolf on clean data is a property of the model rather than
of one split. It is a false-alarm floor and not a detection result: the winner trained on clean labels, so
there is no planted noise in it to find, and no detection recall exists on test.

**Phase B was not attempted.** No K-fold, so no product outside validation has an out-of-sample
prediction.

## Retrieval

Same-product retrieval over the whole catalog, three embedding sources so that fine-tuning is separated
from pretraining corpus and from architecture. The query set is the 38,655 images belonging to the 13,153
products with two or more colorways; the 11,370 single-colorway products are excluded because they have no
relevant item to find. mAP@10, macro over queries, gallery is all 50,025 images.

| source | mAP@10 | over products | R@10 capped | R@10 uncapped | CMC@1 | CMC@10 |
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

C1 is the only clean one: same architecture, same corpus, same pipeline, with fine-tuning the only
difference. C2 moves fine-tuning, corpus and architecture at once and is reported without attribution.
C3 exists to make C2 readable at all.

**A registered prediction was inverted here, and the reason is worth stating.** The project expected
fine-tuning to *lose*, on the reasoning that training on attributes pulls the space toward attribute
clusters and away from instance identity. It did not lose. The ground truth is why: a relevant item is the
same ProductID at a different ColorID, and two colorways share **every one of the eight families**. They
differ only in colour, and colour is not one of them. So an encoder trained on those families is trained
to map colorway siblings to the same point. **Attribute clustering and same-product clustering are the
same clustering here**, which also means this does not generalise to visual similarity retrieval.

**About a fifth of C1 is memorisation.** The gallery includes training images, so most queries are images
the fine-tuned backbone has seen. Products never span splits, so a query's targets are always in its own
split:

| source | pooled | train | validation | test |
|---|---:|---:|---:|---:|
| fine-tuned | 0.6620 | 0.6818 | 0.6126 | 0.6213 |
| CLIP | 0.3564 | 0.3546 | 0.3603 | 0.3608 |
| ImageNet-pretrained | 0.2568 | 0.2537 | 0.2674 | 0.2601 |

C1 is **+0.4281 on seen queries and +0.3452 / +0.3612 on unseen ones**. A control makes that estimate
conservative rather than generous: if the unseen strata were simply harder, every source would drop on
them, and the two that never trained on anything both score slightly *better* there. Only the fine-tuned
source falls.

Both numbers are legitimate and answer different questions. The pooled figure is what the deployed demo
does on this catalog, where most queries genuinely are indexed training images. The unseen figure is what
the representation is worth on products it has never seen.
