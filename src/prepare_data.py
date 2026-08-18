"""Decodes the images once into a memmap and writes the label artifacts every run depends on."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src import catalog, splits
from src.eda import DataPaths

IMAGES_NAME: str = "images_u8.npy"
LABELS_NAME: str = "labels.npy"
OBSERVED_NAME: str = "family_observed.npy"
TARGETS_NAME: str = "softmax_targets.npy"
SCHEMA_NAME: str = "label_schema.json"
SPLITS_NAME: str = "splits.json"
CATALOG_NAME: str = "catalog.csv"
MANIFEST_NAME: str = "manifest.json"
VERIFY_SAMPLE: int = 32
VERIFY_SEED: int = 5


class PreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifacts:
    rows: int
    labels: int
    normalized_images: int
    reused_memmap: bool
    out_dir: Path


def load_expected_survivors(eda_path: Path) -> list[str]:
    if not eda_path.exists():
        raise PreparationError(
            f"{eda_path} not found. Run 'uv run python -m src.eda' before prepare_data so the "
            "surviving-label list can be cross-checked against an independently derived one."
        )
    report = json.loads(eda_path.read_text(encoding="utf-8"))
    surviving = report.get("labels", {}).get("surviving_labels")
    if not surviving:
        raise PreparationError(f"{eda_path} has no labels.surviving_labels; regenerate it with src.eda")
    return [str(name) for name in surviving]


def assert_survivors_match(derived: tuple[str, ...], expected: list[str], eda_path: Path) -> None:
    if list(derived) == expected:
        return
    derived_set, expected_set = set(derived), set(expected)
    only_here = sorted(derived_set - expected_set)
    only_there = sorted(expected_set - derived_set)
    if not only_here and not only_there:
        raise PreparationError(
            f"surviving labels match {eda_path} by name but not by order; the schema ordering diverged"
        )
    raise PreparationError(
        f"surviving labels disagree with {eda_path}: {len(derived)} here vs {len(expected)} there; "
        f"only in prepare_data {only_here[:5]}; only in eda {only_there[:5]}"
    )


def decode_images(square_root: Path, relative_paths: list[str], destination: Path) -> int:
    array = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.uint8,
        shape=(len(relative_paths), catalog.IMAGE_SIZE, catalog.IMAGE_SIZE, 3),
    )
    normalized = 0
    for index, relative in enumerate(tqdm(relative_paths, desc="decoding", unit="img")):
        with Image.open(square_root / relative) as image:
            if image.size != (catalog.IMAGE_SIZE, catalog.IMAGE_SIZE) or image.mode != "RGB":
                normalized += 1
            array[index] = np.asarray(catalog.normalize_image(image), dtype=np.uint8)
    array.flush()
    del array
    return normalized


def verify_memmap(square_root: Path, relative_paths: list[str], destination: Path) -> None:
    array = np.load(destination, mmap_mode="r")
    if array.shape != (len(relative_paths), catalog.IMAGE_SIZE, catalog.IMAGE_SIZE, 3):
        raise PreparationError(f"{destination} has shape {array.shape}, expected {len(relative_paths)} rows")
    rng = np.random.default_rng(VERIFY_SEED)
    sample = rng.choice(len(relative_paths), min(VERIFY_SAMPLE, len(relative_paths)), replace=False)
    for index in sample:
        with Image.open(square_root / relative_paths[int(index)]) as image:
            expected = np.asarray(catalog.normalize_image(image), dtype=np.uint8)
        if not np.array_equal(array[int(index)], expected):
            name = relative_paths[int(index)]
            raise PreparationError(f"row {index} of {destination} does not match {name}")


def existing_memmap_is_reusable(destination: Path, rows: int) -> bool:
    if not destination.exists():
        return False
    array = np.load(destination, mmap_mode="r")
    return bool(array.shape == (rows, catalog.IMAGE_SIZE, catalog.IMAGE_SIZE, 3))


def prepare(
    paths: DataPaths,
    out_dir: Path,
    eda_path: Path,
    threshold: int = catalog.MIN_LABEL_POSITIVES,
    seed: int = splits.DEFAULT_SEED,
    force: bool = False,
) -> Artifacts:
    built = catalog.build_catalog(paths.square, paths.bin_csv)
    frame = built.frame
    groups = frame["product_id"].to_numpy()

    split_indices = splits.make_splits(groups, seed=seed)
    splits.assert_no_product_leakage(groups, split_indices)

    train_counts = catalog.positives_per_label(frame, built.label_columns, split_indices["train"])
    survivors = catalog.surviving_labels(train_counts, threshold)
    schema = catalog.build_label_schema(survivors, built.label_columns)
    assert_survivors_match(schema.columns, load_expected_survivors(eda_path), eda_path)

    labels = catalog.label_matrix(frame, schema.columns).astype(np.uint8)
    splits.assert_test_label_coverage(labels, split_indices["test"], schema.columns)
    observed = catalog.family_observed(frame, survivors)
    targets = catalog.softmax_targets(labels, schema)

    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / IMAGES_NAME
    relative_paths = [str(p) for p in frame["path"]]

    reused = not force and existing_memmap_is_reusable(destination, len(relative_paths))
    normalized = 0 if reused else decode_images(paths.square, relative_paths, destination)
    verify_memmap(paths.square, relative_paths, destination)

    np.save(out_dir / LABELS_NAME, labels)
    np.save(out_dir / OBSERVED_NAME, observed)
    np.save(out_dir / TARGETS_NAME, targets)
    schema.save(out_dir / SCHEMA_NAME)
    splits.save_splits(out_dir / SPLITS_NAME, split_indices, seed=seed)
    frame.loc[:, ["cid", "product_id", "color_id", "dir_cat", "dir_sub", "brand", "path"]].to_csv(
        out_dir / CATALOG_NAME, index=False
    )

    manifest = {
        "rows": int(len(frame)),
        "labels": len(schema.columns),
        "softmax_labels": sum(len(f.labels) for f in schema.softmax_families()),
        "bce_labels": sum(len(f.labels) for f in schema.bce_families()),
        "threshold": threshold,
        "seed": seed,
        "image_size": catalog.IMAGE_SIZE,
        "normalized_images": normalized,
        "reused_memmap": reused,
        "split_counts": {name: int(len(idx)) for name, idx in split_indices.items()},
        "catalog_stats": built.stats,
        "shapes": {
            IMAGES_NAME: [len(frame), catalog.IMAGE_SIZE, catalog.IMAGE_SIZE, 3],
            LABELS_NAME: list(labels.shape),
            OBSERVED_NAME: list(observed.shape),
            TARGETS_NAME: list(targets.shape),
        },
        "survivors_cross_checked_against": str(eda_path),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return Artifacts(
        rows=len(frame),
        labels=len(schema.columns),
        normalized_images=normalized,
        reused_memmap=reused,
        out_dir=out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode images once and write the label artifacts")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--eda", type=Path, default=Path("artifacts/eda.json"))
    parser.add_argument("--threshold", type=int, default=catalog.MIN_LABEL_POSITIVES)
    parser.add_argument("--seed", type=int, default=splits.DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="re-decode even if the memmap already exists")
    args = parser.parse_args()

    result = prepare(
        DataPaths.from_root(args.data_root),
        args.out_dir,
        args.eda,
        args.threshold,
        args.seed,
        args.force,
    )
    action = "reused existing" if result.reused_memmap else f"decoded, {result.normalized_images} normalized"
    print(f"rows      {result.rows}")
    print(f"labels    {result.labels}")
    print(f"images    {action}")
    print(f"written   {result.out_dir}")


if __name__ == "__main__":
    main()
