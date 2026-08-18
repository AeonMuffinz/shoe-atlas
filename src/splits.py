"""Grouped train/val/test split by product, with the two structural assertions that guard it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
DEFAULT_SEED: int = 42
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)


class SplitValidationError(AssertionError):
    pass


def make_splits(
    product_ids: np.ndarray,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, np.ndarray]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    groups = np.asarray(product_ids)
    indices = np.arange(len(groups))

    first = GroupShuffleSplit(n_splits=1, train_size=ratios[0], random_state=seed)
    train_pos, rest_pos = next(first.split(indices, groups=groups))
    rest = indices[rest_pos]

    val_share = ratios[1] / (ratios[1] + ratios[2])
    second = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=seed)
    val_pos, test_pos = next(second.split(rest, groups=groups[rest]))

    return {
        "train": np.sort(indices[train_pos]),
        "val": np.sort(rest[val_pos]),
        "test": np.sort(rest[test_pos]),
    }


def assert_no_product_leakage(product_ids: np.ndarray, splits: dict[str, np.ndarray]) -> None:
    groups = np.asarray(product_ids)
    seen = {name: set(groups[idx].tolist()) for name, idx in splits.items()}
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1 :]:
            shared = seen[left] & seen[right]
            if shared:
                sample = sorted(shared)[:5]
                raise SplitValidationError(
                    f"{len(shared)} product_id(s) appear in both {left} and {right}, e.g. {sample}"
                )


def assert_test_label_coverage(
    labels: np.ndarray,
    test_indices: np.ndarray,
    columns: tuple[str, ...] | list[str],
) -> None:
    positives = labels[test_indices].sum(axis=0)
    empty = [str(columns[i]) for i in np.flatnonzero(positives == 0)]
    if empty:
        raise SplitValidationError(
            f"{len(empty)} surviving label(s) have no positive in test and cannot be scored there: {empty}"
        )


def labels_without_validation_positives(
    labels: np.ndarray,
    val_indices: np.ndarray,
    columns: tuple[str, ...] | list[str],
) -> list[str]:
    positives = labels[val_indices].sum(axis=0)
    return [str(columns[i]) for i in np.flatnonzero(positives == 0)]


def save_splits(
    path: Path,
    splits: dict[str, np.ndarray],
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "ratios": list(ratios),
        "counts": {name: int(len(idx)) for name, idx in splits.items()},
        "indices": {name: [int(i) for i in idx] for name, idx in splits.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_splits(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: np.asarray(idx, dtype=np.int64) for name, idx in payload["indices"].items()}
