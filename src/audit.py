"""The catalog audit: corrupt labels a known amount, run both cleanlab backends, score what was found."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import metrics, reporting
from src.catalog import FAMILIES, LabelSchema

UNIFORM: str = "uniform"
CONFUSION: str = "confusion_weighted"
ADD: str = "add"
DROP: str = "drop"
AUDIT_NAME: str = "audit.json"
IMPUTATION_NAME: str = "imputation.json"
DEFAULT_RATES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
PRECISION_K: int = 100


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Corruption:
    labels: np.ndarray
    corrupted: np.ndarray
    kind: str
    rate: float

    @property
    def count(self) -> int:
        return int(self.corrupted.sum())


def observed_rows(family_observed: np.ndarray, family: str) -> np.ndarray:
    return np.flatnonzero(family_observed[:, FAMILIES.index(family)])


def sample_rows(rows: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 <= rate <= 1.0:
        raise AuditError(f"corruption rate must be a fraction, got {rate}")
    count = int(round(len(rows) * rate))
    return rng.choice(rows, size=count, replace=False) if count else np.empty(0, dtype=np.int64)


def reassign_uniform(current: int, width: int, rng: np.random.Generator) -> int:
    choices = [c for c in range(width) if c != current]
    return int(rng.choice(choices))


def reassign_by_confusion(
    current: int, confusion: np.ndarray, rng: np.random.Generator
) -> int:
    weights = np.array(confusion[current], dtype=np.float64)
    weights[current] = 0.0
    total = weights.sum()
    if total <= 0:
        return reassign_uniform(current, len(weights), rng)
    return int(rng.choice(len(weights), p=weights / total))


def corrupt_exclusive_family(
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    family: str,
    rate: float,
    rng: np.random.Generator,
    confusion: np.ndarray | None = None,
) -> Corruption:
    slice_ = schema.family(family)
    if slice_.kind != "softmax":
        raise AuditError(
            f"{family} is a {slice_.kind} family. Reassignment keeps exactly one positive per row, "
            "which is only meaningful where the classes are mutually exclusive."
        )
    out = labels.copy()
    flagged = np.zeros_like(labels, dtype=bool)
    width = slice_.end - slice_.start
    for row in sample_rows(observed_rows(family_observed, family), rate, rng):
        block = out[row, slice_.start : slice_.end]
        current = int(block.argmax())
        replacement = (
            reassign_uniform(current, width, rng)
            if confusion is None
            else reassign_by_confusion(current, confusion, rng)
        )
        block[:] = 0
        block[replacement] = 1
        flagged[row, slice_.start + current] = True
        flagged[row, slice_.start + replacement] = True
    kind = UNIFORM if confusion is None else CONFUSION
    return Corruption(labels=out, corrupted=flagged, kind=kind, rate=rate)


def corrupt_bce_family(
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    family: str,
    rate: float,
    rng: np.random.Generator,
    mode: str,
) -> Corruption:
    slice_ = schema.family(family)
    if slice_.kind != "bce":
        raise AuditError(f"{family} is exclusive; a bit flip would leave it with zero or two positives")
    if mode not in (ADD, DROP):
        raise AuditError(f"unknown bit-flip mode {mode!r}")
    out = labels.copy()
    flagged = np.zeros_like(labels, dtype=bool)
    rows = observed_rows(family_observed, family)
    wanted = 1.0 if mode == DROP else 0.0
    cells = [
        (row, column)
        for row in rows
        for column in range(slice_.start, slice_.end)
        if out[row, column] == wanted
    ]
    if cells:
        picked = sample_rows(np.arange(len(cells)), rate, rng)
        for index in picked:
            row, column = cells[int(index)]
            out[row, column] = 0.0 if mode == DROP else 1.0
            flagged[row, column] = True
    return Corruption(labels=out, corrupted=flagged, kind=mode, rate=rate)


def multihot_to_index_lists(block: np.ndarray) -> list[list[int]]:
    return [sorted(int(c) for c in np.flatnonzero(row)) for row in block]


def assert_rows_sum_to_one(block: np.ndarray, family: str, tolerance: float = 1e-3) -> None:
    sums = block.sum(axis=1)
    if sums.size and not np.allclose(sums, 1.0, atol=tolerance):
        raise AuditError(
            f"{family} probabilities do not sum to one per row (min {sums.min():.4f}, "
            f"max {sums.max():.4f}). cleanlab's multiclass path assumes a distribution over the family."
        )


def assert_issue_shape(issues: np.ndarray, expected: tuple[int, int], family: str) -> None:
    if issues.shape != expected:
        raise AuditError(
            f"{family}: cleanlab returned issues of shape {issues.shape} but the probability block is "
            f"{expected}. The per-class multilabel filter returns one row per example and one column per "
            "label; reading it the other way round silently transposes the findings whenever the two "
            "happen to match."
        )


def exclusive_issues(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, np.ndarray]:
    from cleanlab.filter import find_label_issues

    flagged: dict[str, np.ndarray] = {}
    for family in schema.softmax_families():
        rows = observed_rows(family_observed, family.name)
        mask = np.zeros(len(labels), dtype=bool)
        if rows.size:
            block = probs[rows, family.start : family.end]
            assert_rows_sum_to_one(block, family.name)
            given = labels[rows, family.start : family.end].argmax(axis=1)
            issues = find_label_issues(labels=given, pred_probs=block, filter_by="prune_by_noise_rate")
            mask[rows] = np.asarray(issues, dtype=bool)
        flagged[family.name] = mask
    return flagged


def bce_issues(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, np.ndarray]:
    from cleanlab.multilabel_classification.filter import find_multilabel_issues_per_class

    flagged: dict[str, np.ndarray] = {}
    for family in schema.bce_families():
        rows = observed_rows(family_observed, family.name)
        mask = np.zeros((len(labels), family.end - family.start), dtype=bool)
        if rows.size:
            block = probs[rows, family.start : family.end]
            given = multihot_to_index_lists(labels[rows, family.start : family.end])
            issues = np.asarray(
                find_multilabel_issues_per_class(
                    labels=given, pred_probs=block, return_indices_ranked_by=None
                ),
                dtype=bool,
            )
            assert_issue_shape(issues, block.shape, family.name)
            mask[rows] = issues
        flagged[family.name] = mask
    return flagged


def detection_metrics(
    flagged: np.ndarray, corrupted: np.ndarray, k: int = PRECISION_K
) -> dict[str, float]:
    truth = corrupted.astype(bool).ravel()
    found = flagged.astype(bool).ravel()
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    hits = int((found & truth).sum())
    false_alarms = int((found & ~truth).sum())
    ranked = np.flatnonzero(found)[:k]
    return {
        "corrupted_cells": positives,
        "flagged_cells": int(found.sum()),
        "recall": hits / positives if positives else float("nan"),
        "false_alarm_rate": false_alarms / negatives if negatives else float("nan"),
        "precision": hits / int(found.sum()) if found.any() else float("nan"),
        "precision_at_k": float(truth[ranked].mean()) if ranked.size else float("nan"),
        "k": int(min(k, ranked.size)),
    }


def unobserved_cells(family_observed: np.ndarray, schema: LabelSchema) -> np.ndarray:
    return ~metrics.cell_mask(family_observed, schema)


def imputation_candidates(
    probs: np.ndarray, family_observed: np.ndarray, schema: LabelSchema, threshold: float = 0.9
) -> dict[str, int]:
    unobserved = unobserved_cells(family_observed, schema)
    confident = unobserved & (probs >= threshold)
    return {
        "unobserved_cells": int(unobserved.sum()),
        "confident_fills": int(confident.sum()),
        "threshold": threshold,
    }


def score_backends(
    probs: np.ndarray,
    corruption: Corruption,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, dict[str, float]]:
    exclusive = exclusive_issues(probs, corruption.labels, family_observed, schema)
    multi = bce_issues(probs, corruption.labels, family_observed, schema)

    per_group: dict[str, dict[str, float]] = {}
    for family in schema.softmax_families():
        rows = exclusive[family.name]
        truth = corruption.corrupted[:, family.start : family.end].any(axis=1)
        per_group[family.name] = detection_metrics(rows, truth)
    for family in schema.bce_families():
        per_group[family.name] = detection_metrics(
            multi[family.name], corruption.corrupted[:, family.start : family.end]
        )
    return per_group


def summarise_groups(per_family: dict[str, dict[str, float]], schema: LabelSchema) -> dict[str, dict]:
    exclusive = [f.name for f in schema.softmax_families()]
    return {
        "exclusive_families": {name: per_family[name] for name in exclusive if name in per_family},
        "bce_families": {
            name: per_family[name] for name in per_family if name not in exclusive
        },
        "note": (
            "detection is reported per family group and never pooled. The multiclass path estimates a "
            "full joint distribution; the multilabel path decomposes to one-against-all and has weaker "
            "joint structure, so a single headline number would average two backends of different "
            "strength and flatter the weaker one."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the catalog audit against known corruption")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=reporting.PROCESSED_DIR)
    args = parser.parse_args()
    raise AuditError(
        f"phase A needs a model trained on corrupted labels, which {args.run} does not yet carry. "
        "The corruption, both cleanlab backends and the detection metrics are importable and tested; "
        "what is missing is a training run over a corrupted label matrix."
    )


if __name__ == "__main__":
    main()
