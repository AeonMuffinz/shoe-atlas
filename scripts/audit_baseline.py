"""Runs both cleanlab backends against clean labels, so the flags they raise with nothing planted are
measured rather than assumed. This is the false-alarm floor the corrupted runs are read against."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import audit, reporting
from src.catalog import LabelSchema
from src.corruption import Corruption

BASELINE_NAME: str = "audit_baseline_0pct_{split}.json"
NOTE: str = (
    "FINDINGS 24.4 baseline. Trained under the constrained selector rather than the fixed epoch, so it is "
    "NOT matched to the six Phase A runs and is not the zero point of a controlled curve. With zero "
    "corruption every flag is a false alarm by construction, so recall is undefined and the false alarm "
    "rate is the whole result: it is what cleanlab flags when there is nothing to find."
)


def load_clean_inputs(
    run_dir: Path, processed: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelSchema]:
    audit.assert_split_auditable(split)
    schema = LabelSchema.load(processed / "label_schema.json")
    indices = json.loads((processed / "splits.json").read_text(encoding="utf-8"))["indices"]
    rows = np.asarray(indices[split], dtype=np.int64)

    probs_path = run_dir / reporting.PROBS_OOF_NAME.format(split=split)
    if not probs_path.exists():
        raise audit.AuditError(
            f"{probs_path.name} is missing. cleanlab needs out-of-sample probabilities; run evaluate.py "
            f"for this run with the {split} split scored before taking a baseline on it."
        )
    probs = np.load(probs_path).astype(np.float64)
    labels = np.load(processed / "labels.npy")[rows].astype(np.float64)
    observed = np.load(processed / "family_observed.npy")[rows]
    return probs, labels, observed, schema


def run_baseline(run_dir: Path, processed: Path, split: str) -> dict[str, object]:
    probs, labels, observed, schema = load_clean_inputs(run_dir, processed, split)
    clean = Corruption(
        labels=labels, corrupted=np.zeros_like(labels, dtype=bool), kind="none", rate=0.0
    )
    per_family = audit.score_backends(probs, clean, observed, schema)
    return {
        "source_run": run_dir.name,
        "split": split,
        "rate": 0.0,
        "note": NOTE,
        "probabilities": reporting.PROBS_OOF_NAME.format(split=split),
        "rows": int(labels.shape[0]),
        "per_family": per_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit false-alarm floor at zero corruption")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=reporting.PROCESSED_DIR)
    parser.add_argument("--split", choices=sorted(audit.AUDITABLE_SPLITS), default="val")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_baseline(args.run, args.processed, args.split)
    destination = args.out or (Path("artifacts") / BASELINE_NAME.format(split=args.split))
    destination.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"baseline written to {destination}")
    for family, scores in report["per_family"].items():
        print(f"  {family:<14} flagged {scores['flagged_cells']:>6}  "
              f"false alarm rate {scores['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    main()
