"""The no-model baseline: predict each label's training frequency and score it like any other run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src import data_setup, metrics, reporting
from src.catalog import LabelSchema
from src.reporting import PROCESSED_DIR, RUNS_ROOT, TEST_WITHHELD

RUN_NAME: str = "prevalence_floor"
FOLDS: int = 5


def constant_predictions(train_scores: np.ndarray, rows: int) -> np.ndarray:
    return np.tile(train_scores, (rows, 1))


def out_of_fold_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    folds: int = FOLDS,
    seed: int = 0,
) -> dict[str, float]:
    index = np.arange(probs.shape[0])
    assignment = np.random.default_rng(seed).permutation(index) % folds
    predicted = np.zeros_like(labels, dtype=np.int8)
    held_out = np.zeros_like(labels, dtype=np.float64)
    for fold in range(folds):
        fit = index[assignment != fold]
        held = index[assignment == fold]
        if fit.size == 0 or held.size == 0:
            continue
        thresholds, _ = metrics.sweep_thresholds(
            probs[fit], labels[fit], metrics.cell_mask(family_observed[fit], schema), schema
        )
        predicted[held] = (probs[held] >= thresholds).astype(np.int8)
        held_out[held] = probs[held]

    mask = metrics.cell_mask(family_observed, schema)
    f1 = np.full(labels.shape[1], np.nan)
    for family in schema.bce_families():
        for column in range(family.start, family.end):
            valid = np.flatnonzero(mask[:, column])
            if valid.size == 0:
                continue
            f1[column] = float(
                f1_score(labels[valid, column], predicted[valid, column], zero_division=0)
            )
    macro_f1, _ = metrics.macro_average(f1)
    calibration, _ = metrics.macro_average(
        metrics.per_label_calibration_error(held_out, labels, mask)
    )
    return {"macro_f1_bce": macro_f1, "calibration_error": calibration}


def build(processed_dir: Path = PROCESSED_DIR, runs_root: Path = RUNS_ROOT) -> dict[str, object]:
    artifacts = data_setup.load_artifacts(processed_dir)
    schema = artifacts.schema
    train, val = artifacts.splits["train"], artifacts.splits["val"]

    train_labels = artifacts.labels[train].astype(np.float64)
    train_mask = metrics.cell_mask(artifacts.family_observed[train], schema)
    scores = metrics.prevalence_scores(train_labels, train_mask)

    labels = artifacts.labels[val].astype(np.float64)
    observed = artifacts.family_observed[val]
    probs = constant_predictions(scores, len(val))
    mask = metrics.cell_mask(observed, schema)

    thresholds, uncalibrated = metrics.sweep_thresholds(probs, labels, mask, schema)
    summary = metrics.summarise(probs, labels, observed, schema, thresholds)
    honest = out_of_fold_scores(probs, labels, observed, schema)

    run_dir = runs_root / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "run": RUN_NAME,
        "stem": RUN_NAME,
        "role": "candidate",
        "checkpoint": None,
        "epoch": None,
        "selection_metric": None,
        "val": {
            "map": summary["map"],
            "map_softmax": summary["map_softmax_labels"],
            "map_bce": summary["map_bce_labels"],
            "map_calibrated": summary["map"],
            "macro_f1_bce": summary["macro_f1_bce"],
            "macro_f1_bce_out_of_fold": honest["macro_f1_bce"],
            "calibration_error": summary["calibration_error"],
            "calibration_error_out_of_fold": honest["calibration_error"],
            "calibration_error_uncalibrated": summary["calibration_error"],
            "family_top1": summary["family_top1"],
            "unscoreable_labels": summary["unscoreable_labels"],
            "uncalibrated_thresholds": uncalibrated,
            "identity_calibrated_labels": [],
            "rows": int(len(val)),
            "note_macro_f1": (
                "a constant predictor scores the same fitted and out of fold, because every row "
                "carries the same score and the best threshold cannot vary by fold"
            ),
            "note_map": (
                "a constant predictor returning each label's training frequency; it is already "
                "calibrated by construction, so calibrated and uncalibrated figures coincide"
            ),
        },
        "test": TEST_WITHHELD,
    }
    reporting.assert_contract(report["val"], RUN_NAME)
    reporting.write_report(run_dir, report)

    run_summary = {
        "name": RUN_NAME,
        "stem": RUN_NAME,
        "role": "candidate",
        "selection_metric": None,
        "selection_scope": "no model is trained, so no epoch is selected",
        "seed": None,
        "epochs_completed": 0,
        "device": "none",
        "wall_clock_seconds": 0.0,
        "note": (
            "predicts each label's training-split frequency; a model has to clear this "
            "before its mAP means anything"
        ),
    }
    reporting.write_summary(run_dir, run_summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the prevalence-floor baseline as a persisted run")
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = parser.parse_args()

    report = build(args.processed, args.runs_root)
    val = report["val"]
    print(f"prevalence floor over {val['rows']} validation rows")
    print(f"  mAP          {val['map']:.4f}   softmax {val['map_softmax']:.4f}   bce {val['map_bce']:.4f}")
    print(f"  macro F1 bce {val['macro_f1_bce']:.4f}   out-of-fold {val['macro_f1_bce_out_of_fold']:.4f}")
    print(
        f"  calibration  {val['calibration_error']:.4f}   "
        f"out-of-fold {val['calibration_error_out_of_fold']:.4f}"
    )
    print(f"  family top-1 {({k: round(v, 4) for k, v in val['family_top1'].items()})}")


if __name__ == "__main__":
    main()
