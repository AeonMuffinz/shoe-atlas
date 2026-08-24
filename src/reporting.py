"""The evaluation.json contract. Every run scores through here so the comparison table stays uniform."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import metrics
from src.catalog import MIN_LABEL_POSITIVES, LabelSchema

RUNS_ROOT: Path = Path("artifacts/runs")
PROCESSED_DIR: Path = Path("data/processed")
WINNER_PATH: Path = Path("artifacts/winner.json")
EVALUATION_NAME: str = "evaluation.json"
SUMMARY_NAME: str = "run_summary.json"
THRESHOLDS_NAME: str = "thresholds.json"
CALIBRATION_NAME: str = "calibration.json"
CONFIG_NAME: str = "config.yaml"
PROBS_NAME: str = "{split}_probs.npy"
PROBS_OOF_NAME: str = "{split}_probs_oof.npy"
CONFUSION_DIR: str = "confusion"
ERRORS_DIR: str = "errors"
TEST_WITHHELD: str = "not evaluated, test opened once for the winner"

NOTE_MACRO_F1: str = (
    "macro_f1_bce is fitted and scored on the same split and is therefore optimistic; "
    "macro_f1_bce_out_of_fold refits thresholds and calibrators per fold and is the honest one."
)
NOTE_MAP: str = (
    "map is computed on uncalibrated probabilities, matching what training selected on. "
    "Calibration does shift it: isotonic collapses scores into ties and temperature scaling "
    "renormalises each row, so neither preserves per-column ranking. map_calibrated is "
    "reported for transparency but is not the comparison number."
)

VAL_CONTRACT_KEYS: frozenset[str] = frozenset(
    {
        "map",
        "map_softmax",
        "map_bce",
        "map_calibrated",
        "macro_f1_bce",
        "macro_f1_bce_out_of_fold",
        "calibration_error",
        "calibration_error_out_of_fold",
        "calibration_error_uncalibrated",
        "family_top1",
        "unscoreable_labels",
        "uncalibrated_thresholds",
        "identity_calibrated_labels",
        "rows",
        "note_macro_f1",
        "note_map",
    }
)


FP32_BYTES: int = 4
OPTIMIZER_MULTIPLE: int = 3
DISK_SAFETY_MARGIN: float = 1.2

ARCHIVED_KEY: str = "archived"
ARCHIVED_REASON_KEY: str = "archived_reason"


def checkpoint_bytes(parameters: int) -> int:
    return parameters * FP32_BYTES


def checkpoint_budget(
    parameters: int, max_epochs: int, margin: float = DISK_SAFETY_MARGIN
) -> int:
    per = checkpoint_bytes(parameters)
    worst_case = per * (max_epochs + 1) + per * OPTIMIZER_MULTIPLE
    return int(worst_case * margin)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def is_archived(summary: dict[str, object]) -> bool:
    return bool(summary.get(ARCHIVED_KEY, False))


def live_runs(summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {name: body for name, body in summaries.items() if not is_archived(body)}


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Predictions:
    logits: np.ndarray
    labels: np.ndarray
    family_observed: np.ndarray
    rows: np.ndarray


def assert_contract(scores: dict[str, object], run_name: str, extra: frozenset[str] | None = None) -> None:
    keys = set(scores)
    missing = VAL_CONTRACT_KEYS - keys
    if missing:
        raise EvaluationError(
            f"{run_name} is missing {len(missing)} contract key(s), which would force compare_runs.py "
            f"to branch on presence: {sorted(missing)}"
        )
    undeclared = keys - VAL_CONTRACT_KEYS - (extra or frozenset())
    if undeclared:
        raise EvaluationError(
            f"{run_name} emits undeclared key(s) {sorted(undeclared)}; add them to the run's declared "
            "extras so a reader can tell a deliberate addition from a typo"
        )


def assert_test_unlocked(run_name: str, winner_path: Path | None = None) -> dict:
    winner_path = WINNER_PATH if winner_path is None else winner_path
    if not winner_path.exists():
        raise EvaluationError(
            f"refusing to open the test set: {winner_path} does not exist. The test set is opened once, "
            "for the winner only, so a winner has to be recorded before any test score is computed."
        )
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    if str(winner.get("winner")) != run_name:
        raise EvaluationError(
            f"refusing to open the test set: {winner_path} names {winner.get('winner')!r}, "
            f"not {run_name!r}. Losing runs never get a test score."
        )
    return winner


def score_split(
    predictions: Predictions,
    schema: LabelSchema,
    min_positives: int = MIN_LABEL_POSITIVES,
    note_map: str = NOTE_MAP,
    note_macro_f1: str = NOTE_MACRO_F1,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, metrics.Calibrators]:
    calibrators = metrics.fit_calibrators(
        predictions.logits, predictions.labels, predictions.family_observed, schema, min_positives
    )
    calibrated = metrics.apply_calibrators(predictions.logits, schema, calibrators)
    mask = metrics.cell_mask(predictions.family_observed, schema)
    thresholds, uncalibrated = metrics.sweep_thresholds(
        calibrated, predictions.labels, mask, schema
    )

    raw = metrics.logits_to_probabilities(predictions.logits, schema)
    metrics.assert_bce_ranking_survives(predictions.logits, raw, schema)
    ranking = metrics.summarise(
        raw, predictions.labels, predictions.family_observed, schema, thresholds
    )
    summary = metrics.summarise(
        calibrated, predictions.labels, predictions.family_observed, schema, thresholds
    )
    out_of_fold = metrics.out_of_fold_scores(
        predictions.logits, predictions.labels, predictions.family_observed, schema,
        min_positives=min_positives,
    )

    scores: dict[str, object] = {
        "map": ranking["map"],
        "map_softmax": ranking["map_softmax_labels"],
        "map_bce": ranking["map_bce_labels"],
        "map_calibrated": summary["map"],
        "macro_f1_bce": summary["macro_f1_bce"],
        "macro_f1_bce_out_of_fold": out_of_fold["macro_f1_bce"],
        "calibration_error": summary["calibration_error"],
        "calibration_error_out_of_fold": out_of_fold["calibration_error"],
        "calibration_error_uncalibrated": ranking["calibration_error"],
        "family_top1": summary["family_top1"],
        "unscoreable_labels": summary["unscoreable_labels"],
        "uncalibrated_thresholds": uncalibrated,
        "identity_calibrated_labels": calibrators.identity,
        "rows": int(len(predictions.rows)),
        "note_macro_f1": note_macro_f1,
        "note_map": note_map,
    }
    return scores, calibrated, thresholds, calibrators


def ranking_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, float]:
    probs = metrics.logits_to_probabilities(logits, schema)
    mask = metrics.cell_mask(family_observed, schema)
    per_label = metrics.per_label_average_precision(probs, labels, mask)
    value, unscoreable = metrics.macro_average(per_label)
    softmax_columns = [c for f in schema.softmax_families() for c in range(f.start, f.end)]
    bce_columns = [c for f in schema.bce_families() for c in range(f.start, f.end)]
    return {
        "map": value,
        "map_softmax": metrics.macro_average(per_label[softmax_columns])[0],
        "map_bce": metrics.macro_average(per_label[bce_columns])[0],
        "unscoreable_labels": float(unscoreable),
    }


def write_confusion(
    run_dir: Path, calibrated: np.ndarray, predictions: Predictions, schema: LabelSchema
) -> None:
    folder = run_dir / CONFUSION_DIR
    folder.mkdir(parents=True, exist_ok=True)
    matrices = metrics.confusion_matrices(
        calibrated, predictions.labels, predictions.family_observed, schema
    )
    for family, matrix in matrices.items():
        names = schema.family(family).labels
        lines = ["true\\predicted," + ",".join(names)]
        for name, row in zip(names, matrix, strict=True):
            lines.append(name + "," + ",".join(str(int(v)) for v in row))
        (folder / f"{family}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(run_dir: Path, report: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / EVALUATION_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_summary(run_dir: Path, summary: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
