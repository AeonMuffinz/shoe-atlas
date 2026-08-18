"""Masked evaluation metrics. Every score sees only the rows where that label's family is observed."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from src.catalog import FAMILIES, LabelSchema, expand_family_mask

DEFAULT_GRID: np.ndarray = np.round(np.arange(0.10, 0.9001, 0.02), 2)
DEFAULT_THRESHOLD: float = 0.5
CALIBRATION_BINS: int = 15


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def logits_to_probabilities(logits: np.ndarray, schema: LabelSchema) -> np.ndarray:
    probs = np.zeros_like(logits, dtype=np.float64)
    for family in schema.families:
        block = logits[:, family.start : family.end]
        probs[:, family.start : family.end] = (
            softmax(block) if family.kind == "softmax" else sigmoid(block)
        )
    return probs


def cell_mask(family_observed: np.ndarray, schema: LabelSchema) -> np.ndarray:
    return expand_family_mask(family_observed.astype(bool), schema)


def per_label_average_precision(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = np.full(probs.shape[1], np.nan, dtype=np.float64)
    for column in range(probs.shape[1]):
        rows = np.flatnonzero(mask[:, column])
        if rows.size == 0:
            continue
        truth = labels[rows, column]
        if truth.min() == truth.max():
            continue
        scores[column] = float(average_precision_score(truth, probs[rows, column]))
    return scores


def macro_average(values: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan"), int(values.size)
    return float(values[finite].mean()), int((~finite).sum())


def per_label_f1(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    scores = np.full(probs.shape[1], np.nan, dtype=np.float64)
    for column in range(probs.shape[1]):
        rows = np.flatnonzero(mask[:, column])
        if rows.size == 0:
            continue
        truth = labels[rows, column]
        predicted = (probs[rows, column] >= thresholds[column]).astype(np.int8)
        scores[column] = float(f1_score(truth, predicted, zero_division=0))
    return scores


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    schema: LabelSchema,
    grid: np.ndarray | None = None,
    default: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, list[str]]:
    steps = DEFAULT_GRID if grid is None else grid
    thresholds = np.full(probs.shape[1], default, dtype=np.float64)
    uncalibrated: list[str] = []
    for family in schema.bce_families():
        for column in range(family.start, family.end):
            rows = np.flatnonzero(mask[:, column])
            truth = labels[rows, column] if rows.size else np.zeros(0)
            if rows.size == 0 or truth.sum() == 0:
                uncalibrated.append(schema.columns[column])
                continue
            candidates = probs[rows, column]
            best_score, best_threshold = -1.0, default
            for step in steps:
                score = f1_score(truth, (candidates >= step).astype(np.int8), zero_division=0)
                if score > best_score:
                    best_score, best_threshold = float(score), float(step)
            thresholds[column] = best_threshold
    return thresholds, uncalibrated


def family_top1_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for family in schema.softmax_families():
        rows = np.flatnonzero(family_observed[:, FAMILIES.index(family.name)])
        if rows.size == 0:
            accuracies[family.name] = float("nan")
            continue
        block_probs = probs[rows, family.start : family.end]
        block_truth = labels[rows, family.start : family.end]
        accuracies[family.name] = float((block_probs.argmax(1) == block_truth.argmax(1)).mean())
    return accuracies


def confusion_matrices(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for family in schema.softmax_families():
        size = len(family.labels)
        matrix = np.zeros((size, size), dtype=np.int64)
        rows = np.flatnonzero(family_observed[:, FAMILIES.index(family.name)])
        if rows.size:
            predicted = probs[rows, family.start : family.end].argmax(1)
            truth = labels[rows, family.start : family.end].argmax(1)
            np.add.at(matrix, (truth, predicted), 1)
        matrices[family.name] = matrix
    return matrices


def per_label_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    errors = np.full(probs.shape[1], np.nan, dtype=np.float64)
    for column in range(probs.shape[1]):
        rows = np.flatnonzero(mask[:, column])
        if rows.size == 0:
            continue
        confidence = probs[rows, column]
        truth = labels[rows, column]
        total = 0.0
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (confidence > lower) & (confidence <= upper)
            if not in_bin.any():
                continue
            total += in_bin.mean() * abs(confidence[in_bin].mean() - truth[in_bin].mean())
        errors[column] = total
    return errors


def prevalence_scores(train_labels: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    scores = np.zeros(train_labels.shape[1], dtype=np.float64)
    for column in range(train_labels.shape[1]):
        rows = np.flatnonzero(train_mask[:, column])
        scores[column] = float(train_labels[rows, column].mean()) if rows.size else 0.0
    return scores


def summarise(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    thresholds: np.ndarray | None = None,
) -> dict[str, object]:
    mask = cell_mask(family_observed, schema)
    average_precision = per_label_average_precision(probs, labels, mask)
    map_value, unscoreable = macro_average(average_precision)

    if thresholds is None:
        thresholds, uncalibrated = sweep_thresholds(probs, labels, mask, schema)
    else:
        uncalibrated = []
    f1 = per_label_f1(probs, labels, mask, thresholds)
    calibration = per_label_calibration_error(probs, labels, mask)

    bce_columns = [c for family in schema.bce_families() for c in range(family.start, family.end)]
    softmax_columns = [c for family in schema.softmax_families() for c in range(family.start, family.end)]
    bce_f1, _ = macro_average(f1[bce_columns])
    bce_map, _ = macro_average(average_precision[bce_columns])
    softmax_map, _ = macro_average(average_precision[softmax_columns])
    calibration_value, _ = macro_average(calibration)

    return {
        "map": map_value,
        "map_softmax_labels": softmax_map,
        "map_bce_labels": bce_map,
        "macro_f1_bce": bce_f1,
        "calibration_error": calibration_value,
        "unscoreable_labels": unscoreable,
        "uncalibrated_labels": uncalibrated,
        "family_top1": family_top1_accuracy(probs, labels, family_observed, schema),
        "per_label": {
            "average_precision": _by_name(schema, average_precision),
            "f1": _by_name(schema, f1),
            "threshold": {name: float(thresholds[i]) for i, name in enumerate(schema.columns)},
            "calibration_error": _by_name(schema, calibration),
        },
    }


def _by_name(schema: LabelSchema, values: np.ndarray) -> dict[str, float | None]:
    return {name: _clean(values[i]) for i, name in enumerate(schema.columns)}


def _clean(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)
