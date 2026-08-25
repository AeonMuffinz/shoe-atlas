"""Injects a known amount of label noise into the catalog, and records exactly what it changed.

Both the audit and the data pipeline need this, and neither may import the other, so the
primitives live here rather than in either entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.catalog import FAMILIES, FamilySlice, LabelSchema

UNIFORM: str = "uniform"
CONFUSION: str = "confusion_weighted"
ADD: str = "add"
DROP: str = "drop"
TYPE_UNIFORM: str = "uniform"
TYPE_CONFUSION: str = "confusion"
CORRUPT_TYPES: tuple[str, ...] = (TYPE_UNIFORM, TYPE_CONFUSION)
EXCLUSIVE_KIND_FOR_TYPE: dict[str, str] = {TYPE_UNIFORM: UNIFORM, TYPE_CONFUSION: CONFUSION}
LABELS_NAME: str = "labels.npy"
MASK_NAME: str = "corruption_mask.npy"
MANIFEST_NAME: str = "corruption.json"
CORRUPTION_ROOT: str = "corruption"
KIND_NAME: str = "corruption_kind.npy"
KIND_NONE: int = 0
KIND_CODES: dict[str, int] = {UNIFORM: 1, CONFUSION: 2, ADD: 3, DROP: 4}
KIND_NAMES: dict[int, str] = {v: k for k, v in KIND_CODES.items()}


class CorruptionError(RuntimeError):
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


@dataclass(frozen=True)
class CorruptedCatalog:
    labels: np.ndarray
    corrupted: np.ndarray
    rate: float
    seed: int
    per_family: dict[str, dict[str, int]]
    kind: np.ndarray
    corrupt_type: str = TYPE_UNIFORM

    @property
    def count(self) -> int:
        return int(self.corrupted.sum())

    def mask_for(self, kind: str) -> np.ndarray:
        if kind not in KIND_CODES:
            raise CorruptionError(f"unknown corruption kind {kind!r}, expected one of {sorted(KIND_CODES)}")
        return self.kind == KIND_CODES[kind]


def observed_rows(family_observed: np.ndarray, family: str) -> np.ndarray:
    return np.flatnonzero(family_observed[:, FAMILIES.index(family)])


def sample_rows(rows: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    if not 0.0 <= rate <= 1.0:
        raise CorruptionError(f"corruption rate must be a fraction, got {rate}")
    count = int(round(len(rows) * rate))
    return rng.choice(rows, size=count, replace=False) if count else np.empty(0, dtype=np.int64)


def reassign_uniform(current: int, width: int, rng: np.random.Generator) -> int:
    choices = [c for c in range(width) if c != current]
    return int(rng.choice(choices))


def reassign_by_confusion(current: int, confusion: np.ndarray, rng: np.random.Generator) -> int:
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
        raise CorruptionError(
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
        raise CorruptionError(
            f"{family} is exclusive; a bit flip would leave it with zero or two positives"
        )
    if mode not in (ADD, DROP):
        raise CorruptionError(f"unknown bit-flip mode {mode!r}")
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


def corrupt_catalog(
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    rate: float,
    seed: int,
    confusions: dict[str, np.ndarray] | None = None,
    corrupt_type: str = TYPE_UNIFORM,
) -> CorruptedCatalog:
    if not 0.0 < rate <= 1.0:
        raise CorruptionError(f"corruption rate must be in (0, 1], got {rate}")
    assert_type_pairs_with_confusions(corrupt_type, confusions, schema)
    rng = np.random.default_rng(seed)
    out = labels.copy()
    flagged = np.zeros_like(labels, dtype=bool)
    kind = np.zeros(labels.shape, dtype=np.int8)
    per_family: dict[str, dict[str, int]] = {}

    for slice_ in schema.softmax_families():
        confusion = (confusions or {}).get(slice_.name)
        step = corrupt_exclusive_family(
            out, family_observed, schema, slice_.name, rate, rng, confusion
        )
        out = step.labels
        flagged |= step.corrupted
        kind[step.corrupted] = KIND_CODES[step.kind]
        per_family[slice_.name] = {step.kind: int(step.corrupted.any(axis=1).sum())}

    for slice_ in schema.bce_families():
        counts: dict[str, int] = {}
        picks: dict[str, np.ndarray] = {}
        for mode in (ADD, DROP):
            step = corrupt_bce_family(
                labels, family_observed, schema, slice_.name, rate, rng, mode
            )
            picks[mode] = step.corrupted
            counts[mode] = step.count
        out[picks[ADD]] = 1.0
        out[picks[DROP]] = 0.0
        flagged |= picks[ADD] | picks[DROP]
        kind[picks[ADD]] = KIND_CODES[ADD]
        kind[picks[DROP]] = KIND_CODES[DROP]
        per_family[slice_.name] = counts

    assert_exclusive_families_intact(out, family_observed, schema)
    assert_mask_matches_difference(labels, out, flagged)
    assert_kind_covers_mask(flagged, kind)
    assert_exclusive_kind_matches_type(kind, schema, corrupt_type)
    return CorruptedCatalog(
        labels=out,
        corrupted=flagged,
        rate=rate,
        seed=seed,
        per_family=per_family,
        kind=kind,
        corrupt_type=corrupt_type,
    )


def assert_mask_matches_difference(
    clean: np.ndarray, corrupted: np.ndarray, mask: np.ndarray
) -> None:
    changed = corrupted != clean
    if not np.array_equal(changed, mask):
        stale = int((mask & ~changed).sum())
        missed = int((changed & ~mask).sum())
        raise CorruptionError(
            f"the corruption mask disagrees with the labels it describes: {stale} cells flagged but "
            f"unchanged, {missed} changed but unflagged. The mask is the audit's ground truth, so a "
            "cell that was flipped twice back to its original value must not be reported as corrupted."
        )


def assert_exclusive_families_intact(
    labels: np.ndarray, family_observed: np.ndarray, schema: LabelSchema
) -> None:
    for slice_ in schema.softmax_families():
        rows = observed_rows(family_observed, slice_.name)
        if rows.size == 0:
            continue
        totals = labels[rows, slice_.start : slice_.end].sum(axis=1)
        if not np.all(totals == 1):
            bad = int((totals != 1).sum())
            raise CorruptionError(
                f"{slice_.name} has {bad} observed rows without exactly one positive after "
                "corruption. Reassignment must move the positive, never add or remove one."
            )


def corruption_dir(processed_dir: Path, rate: float, seed: int, corrupt_type: str) -> Path:
    assert_known_type(corrupt_type)
    return processed_dir / CORRUPTION_ROOT / f"r{rate:.4f}_s{seed}_{corrupt_type}"


def write_corruption(destination: Path, corrupted: CorruptedCatalog) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / LABELS_NAME, corrupted.labels)
    np.save(destination / MASK_NAME, corrupted.corrupted)
    np.save(destination / KIND_NAME, corrupted.kind)
    manifest = {
        "rate": corrupted.rate,
        "seed": corrupted.seed,
        "corrupt_type": corrupted.corrupt_type,
        "cells_corrupted": corrupted.count,
        "rows_touched": int(corrupted.corrupted.any(axis=1).sum()),
        "per_family": corrupted.per_family,
        "cells_by_kind": {
            name: int((corrupted.kind == code).sum()) for name, code in KIND_CODES.items()
        },
        "kind_codes": KIND_CODES,
        "note": (
            "labels.npy here is the corrupted matrix the model trains on; corruption_mask.npy is the "
            "ground truth the audit scores against. The uncorrupted matrix stays in the parent "
            "directory and is never overwritten."
        ),
    }
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def load_corruption(destination: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    for name in (LABELS_NAME, MASK_NAME, KIND_NAME, MANIFEST_NAME):
        if not (destination / name).exists():
            raise CorruptionError(
                f"{destination} is missing {name}. Generate it with "
                "python -m src.prepare_data --corrupt-rate R --corrupt-seed S --corrupt-type TYPE"
            )
    labels = np.load(destination / LABELS_NAME)
    mask = np.load(destination / MASK_NAME)
    kind = np.load(destination / KIND_NAME)
    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert_kind_covers_mask(mask, kind)
    manifest["kind"] = kind
    return labels, mask, manifest


def assert_shapes_match(clean: np.ndarray, corrupted: np.ndarray, mask: np.ndarray) -> None:
    if corrupted.shape != clean.shape:
        raise CorruptionError(
            f"corrupted labels are {corrupted.shape} against the clean {clean.shape}; the corruption "
            "set was generated from a different label schema and must be regenerated"
        )
    if mask.shape != clean.shape:
        raise CorruptionError(f"corruption mask is {mask.shape} against labels {clean.shape}")


def assert_kind_covers_mask(mask: np.ndarray, kind: np.ndarray) -> None:
    flagged = mask.astype(bool)
    typed = kind != KIND_NONE
    if not np.array_equal(flagged, typed):
        untyped = int((flagged & ~typed).sum())
        orphan = int((typed & ~flagged).sum())
        raise CorruptionError(
            f"the per-cell corruption kind does not cover the mask: {untyped} corrupted cells carry no "
            f"kind, {orphan} cells carry a kind but are not corrupted. The audit reports detection per "
            "corruption type, so every flagged cell must say which type it is."
        )


def assert_known_type(corrupt_type: str) -> None:
    if corrupt_type not in CORRUPT_TYPES:
        raise CorruptionError(
            f"unknown corruption type {corrupt_type!r}, expected one of {list(CORRUPT_TYPES)}"
        )


def assert_type_pairs_with_confusions(
    corrupt_type: str, confusions: dict[str, np.ndarray] | None, schema: LabelSchema
) -> None:
    assert_known_type(corrupt_type)
    supplied = set(confusions or {})
    wanted = {slice_.name for slice_ in schema.softmax_families()}
    if corrupt_type == TYPE_CONFUSION and supplied < wanted:
        raise CorruptionError(
            f"corrupt_type is {TYPE_CONFUSION!r} but confusion matrices were supplied for "
            f"{sorted(supplied)} against the {len(wanted)} exclusive families {sorted(wanted)}. A "
            "missing matrix falls back to uniform reassignment for that family, which would plant "
            "uniform noise under a confusion-weighted label and make the two indistinguishable."
        )
    if corrupt_type == TYPE_UNIFORM and supplied:
        raise CorruptionError(
            f"corrupt_type is {TYPE_UNIFORM!r} but confusion matrices were supplied for "
            f"{sorted(supplied)}. Uniform corruption must not consult a confusion matrix."
        )


def assert_exclusive_kind_matches_type(
    kind: np.ndarray, schema: LabelSchema, corrupt_type: str
) -> None:
    expected = EXCLUSIVE_KIND_FOR_TYPE[corrupt_type]
    for slice_ in schema.softmax_families():
        block = kind[:, slice_.start : slice_.end]
        planted = {KIND_NAMES[code] for code in np.unique(block) if code != KIND_NONE}
        if planted - {expected}:
            raise CorruptionError(
                f"{slice_.name} was corrupted as {sorted(planted)} but corrupt_type is "
                f"{corrupt_type!r}, which plants {expected!r}."
            )


def assert_type_matches(manifest: dict, requested: str, source: Path) -> None:
    assert_known_type(requested)
    planted = manifest.get("corrupt_type")
    if planted is None:
        raise CorruptionError(
            f"{source} carries no corrupt_type. It was written before the type was recorded and "
            "cannot be shown to hold the scheme this run asks for; regenerate it with "
            "python -m src.prepare_data --corrupt-type TYPE."
        )
    if planted != requested:
        raise CorruptionError(
            f"{source} holds {planted!r} corruption but this run asks for {requested!r}. Detection "
            "recall is reported per corruption type, so training on one and scoring it as the other "
            "would compare two schemes that are the same noise."
        )


def load_confusion(path: Path, family: FamilySlice) -> np.ndarray:
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = rows[0][1:]
    expected = list(family.labels)
    if header != expected:
        raise CorruptionError(
            f"{path.name} column order is {header[:3]}… but {family.name}'s schema order is "
            f"{expected[:3]}…. reassign_by_confusion indexes by within-family position, so a mismatched "
            "column order would silently reassign to the wrong class. Regenerate the confusion matrix "
            "against the current schema rather than reordering it here."
        )
    body = rows[1:]
    if [r[0] for r in body] != expected:
        raise CorruptionError(
            f"{path.name} row order does not match {family.name}'s schema order; the matrix must be "
            "square and indexed identically on both axes."
        )
    matrix = np.array([[float(v) for v in r[1:]] for r in body], dtype=np.float64)
    if matrix.shape != (len(expected), len(expected)):
        raise CorruptionError(f"{path.name} is {matrix.shape}, expected {(len(expected),) * 2}")
    return matrix


def load_confusions(run_dir: Path, schema: LabelSchema) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for family in schema.softmax_families():
        path = run_dir / "confusion" / f"{family.name}.csv"
        if path.exists():
            out[family.name] = load_confusion(path, family)
    return out
