"""Masked evaluation metrics. Every score sees only the rows where that label's family is observed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from src.catalog import FAMILIES, MIN_LABEL_POSITIVES, LabelSchema, expand_family_mask

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


def tie_inflating_bce_columns(
    raw: np.ndarray,
    probs: np.ndarray,
    schema: LabelSchema,
) -> list[str]:
    inflated: list[str] = []
    for family in schema.bce_families():
        for column in range(family.start, family.end):
            if np.unique(probs[:, column]).size < np.unique(raw[:, column]).size:
                inflated.append(schema.columns[column])
    return inflated


def assert_bce_ranking_survives(
    raw: np.ndarray,
    probs: np.ndarray,
    schema: LabelSchema,
) -> None:
    inflated = tie_inflating_bce_columns(raw, probs, schema)
    if inflated:
        raise ValueError(
            f"{len(inflated)} BCE column(s) lost distinct values when scores became probabilities, "
            f"which silently corrupts per-label ranking and therefore mAP: {inflated[:8]}"
        )


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


@dataclass(frozen=True)
class Calibrators:
    temperatures: dict[str, float]
    isotonic: dict[str, tuple[list[float], list[float]]]
    identity: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "temperatures": self.temperatures,
            "isotonic": {k: {"x": x, "y": y} for k, (x, y) in self.isotonic.items()},
            "identity": self.identity,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Calibrators:
        isotonic = {k: (v["x"], v["y"]) for k, v in payload.get("isotonic", {}).items()}
        return cls(
            temperatures={k: float(v) for k, v in payload.get("temperatures", {}).items()},
            isotonic=isotonic,
            identity=list(payload.get("identity", [])),
        )


def temperature_nll(logits: np.ndarray, target: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    log_prob = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    return float(-log_prob[np.arange(len(target)), target].mean())


def fit_temperature(logits: np.ndarray, target: np.ndarray, grid: np.ndarray | None = None) -> float:
    if len(target) == 0:
        return 1.0
    steps = np.geomspace(0.05, 20.0, 200) if grid is None else grid
    losses = [temperature_nll(logits, target, float(t)) for t in steps]
    return float(steps[int(np.argmin(losses))])


def fit_calibrators(
    logits: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    min_positives: int = MIN_LABEL_POSITIVES,
) -> Calibrators:
    from sklearn.isotonic import IsotonicRegression

    temperatures: dict[str, float] = {}
    for family in schema.softmax_families():
        rows = np.flatnonzero(family_observed[:, FAMILIES.index(family.name)])
        block = logits[rows, family.start : family.end]
        target = labels[rows, family.start : family.end].argmax(axis=1)
        temperatures[family.name] = fit_temperature(block, target)

    isotonic: dict[str, tuple[list[float], list[float]]] = {}
    identity: list[str] = []
    mask = cell_mask(family_observed, schema)
    probs = sigmoid(logits)
    for family in schema.bce_families():
        for column in range(family.start, family.end):
            name = schema.columns[column]
            rows = np.flatnonzero(mask[:, column])
            truth = labels[rows, column]
            if rows.size == 0 or truth.sum() < min_positives:
                identity.append(name)
                continue
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(probs[rows, column], truth)
            isotonic[name] = (
                [float(v) for v in model.X_thresholds_],
                [float(v) for v in model.y_thresholds_],
            )
    return Calibrators(temperatures=temperatures, isotonic=isotonic, identity=identity)


def apply_calibrators(logits: np.ndarray, schema: LabelSchema, calibrators: Calibrators) -> np.ndarray:
    out = np.zeros_like(logits, dtype=np.float64)
    for family in schema.softmax_families():
        temperature = calibrators.temperatures.get(family.name, 1.0)
        out[:, family.start : family.end] = softmax(logits[:, family.start : family.end] / temperature)
    for family in schema.bce_families():
        for column in range(family.start, family.end):
            raw = sigmoid(logits[:, column])
            knots = calibrators.isotonic.get(schema.columns[column])
            out[:, column] = raw if knots is None else np.interp(raw, knots[0], knots[1])
    return out


def out_of_fold_scores(
    logits: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    folds: int = 5,
    seed: int = 0,
    min_positives: int = MIN_LABEL_POSITIVES,
) -> dict[str, float]:
    rows = np.arange(logits.shape[0])
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(rows) % folds
    predicted = np.zeros_like(labels, dtype=np.int8)
    calibrated = np.zeros_like(labels, dtype=np.float64)

    for fold in range(folds):
        fit_rows = rows[assignment != fold]
        score_rows = rows[assignment == fold]
        if fit_rows.size == 0 or score_rows.size == 0:
            continue
        calibrators = fit_calibrators(
            logits[fit_rows], labels[fit_rows], family_observed[fit_rows], schema, min_positives
        )
        fit_probs = apply_calibrators(logits[fit_rows], schema, calibrators)
        thresholds, _ = sweep_thresholds(
            fit_probs, labels[fit_rows], cell_mask(family_observed[fit_rows], schema), schema
        )
        held = apply_calibrators(logits[score_rows], schema, calibrators)
        calibrated[score_rows] = held
        predicted[score_rows] = (held >= thresholds).astype(np.int8)

    mask = cell_mask(family_observed, schema)
    f1 = np.full(labels.shape[1], np.nan)
    for family in schema.bce_families():
        for column in range(family.start, family.end):
            valid = np.flatnonzero(mask[:, column])
            if valid.size == 0:
                continue
            f1[column] = float(f1_score(labels[valid, column], predicted[valid, column], zero_division=0))
    macro_f1, _ = macro_average(f1)
    calibration, _ = macro_average(per_label_calibration_error(calibrated, labels, mask))
    return {"macro_f1_bce": macro_f1, "calibration_error": calibration}


CAPPED: str = "capped"
UNCAPPED: str = "uncapped"


class RetrievalDefinitionError(RuntimeError):
    pass


def ranked_relevance(similarity: np.ndarray, relevant: np.ndarray, query: int) -> np.ndarray:
    scores = np.asarray(similarity, dtype=np.float64).copy()
    scores[query] = -np.inf
    order = np.argsort(-scores, kind="stable")
    order = order[order != query]
    return np.asarray(relevant, dtype=bool)[order]


def recall_at_k(hits: np.ndarray, total_relevant: int, k: int, convention: str) -> float:
    if convention not in (CAPPED, UNCAPPED):
        raise RetrievalDefinitionError(
            f"recall@k needs an explicit convention, got {convention!r}. 'capped' divides by "
            "min(k, R) and can reach 1.0; 'uncapped' divides by R and cannot when R > k. They are "
            "different numbers and the choice must be stated, not defaulted."
        )
    if total_relevant <= 0:
        return float("nan")
    found = int(np.asarray(hits, dtype=bool)[:k].sum())
    denominator = min(k, total_relevant) if convention == CAPPED else total_relevant
    return found / denominator


def cmc_at_k(hits: np.ndarray, k: int) -> float:
    return float(bool(np.asarray(hits, dtype=bool)[:k].any()))


def average_precision_at_rank(hits: np.ndarray, total_relevant: int) -> float:
    truth = np.asarray(hits, dtype=bool)
    if total_relevant <= 0:
        return float("nan")
    positions = np.flatnonzero(truth)
    if positions.size == 0:
        return 0.0
    precisions = (np.arange(positions.size) + 1) / (positions + 1)
    return float(precisions.sum() / total_relevant)


def reciprocal_rank(hits: np.ndarray) -> float:
    positions = np.flatnonzero(np.asarray(hits, dtype=bool))
    return float(1.0 / (positions[0] + 1)) if positions.size else 0.0


def retrieval_scores(
    similarity: np.ndarray,
    product_ids: np.ndarray,
    queries: np.ndarray,
    ks: tuple[int, ...],
    convention: str,
) -> dict[str, object]:
    ids = np.asarray(product_ids)
    per_query: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    per_query.update({f"cmc@{k}": [] for k in ks})
    per_query["ap"] = []
    per_query["rr"] = []
    relevant_counts: list[int] = []

    for query in np.asarray(queries, dtype=np.int64):
        relevant = ids == ids[query]
        total = int(relevant.sum()) - 1
        relevant_counts.append(total)
        hits = ranked_relevance(similarity[query], relevant, int(query))
        for k in ks:
            per_query[f"recall@{k}"].append(recall_at_k(hits, total, k, convention))
            per_query[f"cmc@{k}"].append(cmc_at_k(hits, k))
        per_query["ap"].append(average_precision_at_rank(hits, total))
        per_query["rr"].append(reciprocal_rank(hits))

    counts = np.asarray(relevant_counts, dtype=np.float64)
    out: dict[str, object] = {
        name: float(np.nanmean(values)) for name, values in per_query.items()
    }
    out["mean_average_precision"] = out.pop("ap")
    out["mean_reciprocal_rank"] = out.pop("rr")
    out["queries"] = float(len(relevant_counts))
    out["relevant_per_query"] = {
        "mean": float(counts.mean()) if counts.size else float("nan"),
        "median": float(np.median(counts)) if counts.size else float("nan"),
        "min": float(counts.min()) if counts.size else float("nan"),
        "max": float(counts.max()) if counts.size else float("nan"),
    }
    out["recall_convention"] = convention
    return out
