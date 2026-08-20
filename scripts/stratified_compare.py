"""Compares two runs' per-label AP by frequency stratum, floor-corrected, per head group.

Tests whether a capacity gain concentrates where the data is or reaches the frequency-starved tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.error_analysis import (
    STRATA,
    floor_ap,
    floor_relative,
    group_of,
    per_label_ap,
    stratum_of,
)
from src.catalog import LabelSchema

PROCESSED = Path("data/processed")
RUNS = Path("artifacts/runs")
NOISE_FLOOR = 0.001


def load_run(run: str, schema: LabelSchema, val: np.ndarray, labels: np.ndarray,
             observed: np.ndarray) -> dict[str, float]:
    probs = np.load(RUNS / run / "val_probs.npy").astype(np.float64)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"{run} has {probs.shape[0]} rows, the validation split has {labels.shape[0]}")
    return per_label_ap(probs, labels, observed, schema)


def compare(
    baseline: dict[str, float],
    candidate: dict[str, float],
    floor: dict[str, float],
    counts: dict[str, int],
    schema: LabelSchema,
) -> dict[str, dict[str, float]]:
    base_rel = floor_relative(baseline, floor)
    cand_rel = floor_relative(candidate, floor)
    out: dict[str, dict[str, float]] = {}
    for stratum, _low, _high in STRATA:
        names = [
            n for n in baseline
            if stratum_of(counts[n]) == stratum
            and np.isfinite(base_rel[n]["normalised"])
            and np.isfinite(cand_rel[n]["normalised"])
        ]
        entry: dict[str, float] = {"labels": float(len(names))}
        if names:
            for group in ("all", "exclusive", "bce"):
                subset = [n for n in names if group == "all" or group_of(n, schema) == group]
                if not subset:
                    entry[f"delta_{group}"] = float("nan")
                    entry[f"n_{group}"] = 0.0
                    continue
                deltas = [cand_rel[n]["normalised"] - base_rel[n]["normalised"] for n in subset]
                entry[f"delta_{group}"] = float(np.mean(deltas))
                entry[f"n_{group}"] = float(len(subset))
                entry[f"moved_{group}"] = float(sum(abs(d) > NOISE_FLOOR for d in deltas))
        out[stratum] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified floor-corrected comparison of two runs")
    parser.add_argument("--candidate", type=str, default="convnext_base_s42")
    parser.add_argument("--baseline", type=str, default="convnext_tiny_s42")
    args = parser.parse_args()

    schema = LabelSchema.load(PROCESSED / "label_schema.json")
    labels_all = np.load(PROCESSED / "labels.npy")
    observed_all = np.load(PROCESSED / "family_observed.npy")
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))["indices"]
    val = np.asarray(splits["val"], dtype=np.int64)
    train = np.asarray(splits["train"], dtype=np.int64)
    labels = labels_all[val].astype(np.float64)
    observed = observed_all[val]

    baseline = load_run(args.baseline, schema, val, labels, observed)
    candidate = load_run(args.candidate, schema, val, labels, observed)
    floor = floor_ap(
        labels, observed, schema, labels_all[train].astype(np.float64), observed_all[train]
    )
    counts = {
        k: int(v)
        for k, v in json.loads(Path("artifacts/eda.json").read_text(encoding="utf-8"))["labels"][
            "train_positives"
        ].items()
    }

    table = compare(baseline, candidate, floor, counts, schema)
    print(f"{args.candidate} against {args.baseline}, normalised AP, floor-corrected")
    print(f"per-label moves under {NOISE_FLOOR} are inside the noise floor and are not findings\n")
    print(f"{'stratum':<20}{'labels':>7}{'all':>10}{'exclusive':>12}{'bce':>10}{'moved':>8}")
    for stratum, row in table.items():
        print(
            f"{stratum:<20}{row['labels']:>7.0f}"
            f"{row.get('delta_all', float('nan')):>+10.4f}"
            f"{row.get('delta_exclusive', float('nan')):>+12.4f}"
            f"{row.get('delta_bce', float('nan')):>+10.4f}"
            f"{row.get('moved_all', 0):>8.0f}"
        )

    common = table[STRATA[0][0]].get("delta_bce", float("nan"))
    rare = table[STRATA[2][0]].get("delta_bce", float("nan"))
    print("\nverdict against the prediction in FINDINGS 7B.1:")
    if np.isfinite(common) and np.isfinite(rare):
        if rare > NOISE_FLOOR:
            print("  the rare stratum moved: the data-volume reading is FALSIFIED, capacity reaches")
            print("  labels that were supposed to be frequency-limited")
        elif common > NOISE_FLOOR:
            print("  the gain is confined to the common stratum: the data-volume reading SURVIVES")
        else:
            print("  neither stratum moved beyond the noise floor: no capacity effect to attribute")


if __name__ == "__main__":
    main()
