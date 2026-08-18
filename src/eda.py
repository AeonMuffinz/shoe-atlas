"""Measures the dataset facts the design depends on and writes them to artifacts/eda.json."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import catalog, splits

SURVIVAL_THRESHOLDS: tuple[int, ...] = (10, 25, 50, 100, 200, 500)
PADDING_SEED: int = 11
JPEG_NOISE_CEILING: float = 4.0


def catalog_section(built: catalog.Catalog) -> dict[str, object]:
    return dict(built.stats)


def geometry_section(image_root: Path, joined_paths: set[str]) -> dict[str, object]:
    census: Counter[tuple[tuple[int, int], str]] = Counter()
    odd_joined: list[dict[str, object]] = []
    for path in sorted(image_root.rglob("*.jpg")):
        with Image.open(path) as image:
            key = (image.size, image.mode)
        census[key] += 1
        relative = path.relative_to(image_root).as_posix()
        if key != ((catalog.IMAGE_SIZE, catalog.IMAGE_SIZE), "RGB"):
            odd_joined.append(
                {"path": relative, "size": list(key[0]), "mode": key[1], "in_join": relative in joined_paths}
            )
    return {
        "census": [
            {"size": list(size), "mode": mode, "count": count} for (size, mode), count in census.most_common()
        ],
        "non_conforming": odd_joined,
        "non_conforming_total": len(odd_joined),
        "non_conforming_in_join": sum(1 for row in odd_joined if row["in_join"]),
    }


def padding_section(square_root: Path, original_root: Path, samples: int) -> dict[str, object]:
    if not original_root.exists():
        return {"available": False, "reason": "non-square variant not present"}

    files = sorted(square_root.rglob("*.jpg"))
    rng = np.random.default_rng(PADDING_SEED)
    chosen = [files[i] for i in rng.choice(len(files), min(samples, len(files)), replace=False)]

    offsets: list[int] = []
    residuals: list[float] = []
    shrunk = 0
    for path in chosen:
        counterpart = original_root / path.relative_to(square_root)
        if not counterpart.exists():
            continue
        with Image.open(path) as image:
            square = np.asarray(image.convert("L")).astype(np.int16)
        with Image.open(counterpart) as image:
            original = np.asarray(image.convert("L")).astype(np.int16)
        if square.shape[0] < original.shape[0] or square.shape[1] < original.shape[1]:
            shrunk += 1
            continue
        if square.shape[1] != original.shape[1]:
            continue
        span = square.shape[0] - original.shape[0]
        errors = [float(np.abs(square[k : k + original.shape[0]] - original).mean()) for k in range(span + 1)]
        best = int(np.argmin(errors))
        offsets.append(best)
        residuals.append(errors[best])

    counts = Counter(offsets)
    return {
        "available": True,
        "pairs": len(offsets),
        "cropped_cases": shrunk,
        "verdict": "padded" if shrunk == 0 else "mixed",
        "offset_top_distribution": {str(k): v for k, v in sorted(counts.items())},
        "offset_top": int(counts.most_common(1)[0][0]) if counts else None,
        "residual_mae_median": round(float(np.median(residuals)), 3) if residuals else None,
        "residual_within_jpeg_noise": bool(np.max(residuals) < JPEG_NOISE_CEILING) if residuals else None,
    }


def families_section(built: catalog.Catalog, raw_csv: Path) -> dict[str, object]:
    frame = built.frame
    grouped = catalog.family_columns(built.label_columns)
    raw = pd.read_csv(raw_csv, low_memory=False)
    raw["CID"] = raw["CID"].astype(str).str.strip()
    raw = raw[raw["CID"].isin(set(frame["cid"]))]

    rows: list[dict[str, object]] = []
    for family in catalog.FAMILIES:
        block = frame.loc[:, grouped[family]].to_numpy(dtype=np.int16)
        totals = block.sum(axis=1)
        observed = int((totals >= 1).sum())
        multi = int((totals > 1).sum())
        column = raw[family].fillna("").astype(str)
        semicolons = int(column.str.contains(";").sum())
        rows.append(
            {
                "name": family,
                "kind": catalog.family_kind(family),
                "labels": len(grouped[family]),
                "observed": observed,
                "observed_pct": round(100 * observed / len(frame), 2),
                "unobserved": int((totals == 0).sum()),
                "multi_positive": multi,
                "pct_of_observed": round(100 * multi / observed, 2) if observed else 0.0,
                "max_positives": int(totals.max()),
                "raw_semicolon_rows": semicolons,
                "cross_check_ok": bool(semicolons == multi),
            }
        )
    return {"families": rows, "cross_check_all_ok": all(bool(r["cross_check_ok"]) for r in rows)}


def colorways_section(frame: pd.DataFrame) -> dict[str, object]:
    per_product = frame.groupby("product_id").size()
    histogram = per_product.value_counts().sort_index()
    single = int((per_product == 1).sum())
    usable = int(per_product[per_product > 1].sum())
    return {
        "basis": "joined deduplicated rows",
        "images": int(len(frame)),
        "products": int(len(per_product)),
        "mean": round(float(per_product.mean()), 3),
        "median": int(per_product.median()),
        "max": int(per_product.max()),
        "single_colorway_products": single,
        "single_colorway_pct": round(100 * single / len(per_product), 2),
        "multi_colorway_products": int((per_product > 1).sum()),
        "usable_query_images": usable,
        "usable_query_pct": round(100 * usable / len(frame), 2),
        "histogram": {str(k): int(v) for k, v in histogram.items() if k <= 10},
        "over_ten_colorways": int(histogram[histogram.index > 10].sum()),
    }


def directory_section(built: catalog.Catalog) -> dict[str, object]:
    frame = built.frame
    grouped = catalog.family_columns(built.label_columns)
    result: dict[str, object] = {"basis": "joined deduplicated rows", "rows": int(len(frame))}
    for family, column in (("Category", "dir_cat"), ("SubCategory", "dir_sub")):
        names = {c: c.split(".", 1)[1] for c in grouped[family]}
        from_csv = frame.loc[:, grouped[family]].idxmax(axis=1).map(names)
        normalise = lambda s: s.astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.lower()  # noqa: E731
        agree = normalise(frame[column]) == normalise(from_csv)
        result[family] = {
            "agree": int(agree.sum()),
            "disagree": int((~agree).sum()),
            "independent_signal": bool((~agree).any()),
        }
    return result


def splits_section(frame: pd.DataFrame, seed: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    groups = frame["product_id"].to_numpy()
    result = splits.make_splits(groups, seed=seed)
    splits.assert_no_product_leakage(groups, result)
    summary = {
        "seed": seed,
        "counts": {name: int(len(idx)) for name, idx in result.items()},
        "products": {name: int(len(set(groups[idx].tolist()))) for name, idx in result.items()},
        "leakage_assertion": "passed",
    }
    return result, summary


def labels_section(
    built: catalog.Catalog,
    split_indices: dict[str, np.ndarray],
    threshold: int,
) -> dict[str, object]:
    train_counts = catalog.positives_per_label(built.frame, built.label_columns, split_indices["train"])
    curve = {str(t): int((train_counts >= t).sum()) for t in SURVIVAL_THRESHOLDS}
    survivors = catalog.surviving_labels(train_counts, threshold)
    dropped = [c for c in built.label_columns if c not in set(survivors)]
    schema = catalog.build_label_schema(survivors, built.label_columns)

    matrix = catalog.label_matrix(built.frame, schema.columns)
    splits.assert_test_label_coverage(matrix, split_indices["test"], schema.columns)
    uncalibrated = splits.labels_without_validation_positives(
        matrix, split_indices["val"], schema.columns
    )

    val_counts = pd.Series(matrix[split_indices["val"]].sum(axis=0), index=list(schema.columns))
    by_family = {f.name: {"kind": f.kind, "surviving": len(f.labels)} for f in schema.families}
    observed_before = catalog.family_observed(built.frame, built.label_columns)
    observed_after = catalog.family_observed(built.frame, survivors)
    observation_loss = {
        family: {
            "before": int(observed_before[:, i].sum()),
            "after": int(observed_after[:, i].sum()),
            "lost": int(observed_before[:, i].sum() - observed_after[:, i].sum()),
            "kind": catalog.family_kind(family),
        }
        for i, family in enumerate(catalog.FAMILIES)
    }

    return {
        "threshold": threshold,
        "total_labels": len(built.label_columns),
        "surviving": len(survivors),
        "softmax_labels": sum(len(f.labels) for f in schema.softmax_families()),
        "bce_labels": sum(len(f.labels) for f in schema.bce_families()),
        "survival_curve": curve,
        "zero_train_positives": [str(c) for c in train_counts.index if train_counts[c] == 0],
        "surviving_labels": list(schema.columns),
        "dropped": dropped,
        "by_family": by_family,
        "train_positives": {str(k): int(v) for k, v in train_counts.loc[list(schema.columns)].items()},
        "val_positives": {str(k): int(v) for k, v in val_counts.items()},
        "test_positives": "withheld until the test unlock; coverage checked by assertion only",
        "test_coverage_assertion": "passed",
        "uncalibrated_labels": uncalibrated,
        "observation_loss_from_filtering": observation_loss,
    }


@dataclass(frozen=True)
class DataPaths:
    square: Path
    original: Path
    bin_csv: Path
    raw_csv: Path

    @classmethod
    def from_root(cls, root: Path) -> DataPaths:
        meta = root / "ut-zap50k-data"
        return cls(
            square=root / "ut-zap50k-images-square",
            original=root / "ut-zap50k-images",
            bin_csv=meta / "meta-data-bin.csv",
            raw_csv=meta / "meta-data.csv",
        )


def run(
    paths: DataPaths,
    out_path: Path | None = None,
    threshold: int = catalog.MIN_LABEL_POSITIVES,
    seed: int = splits.DEFAULT_SEED,
    padding_samples: int = 250,
) -> dict[str, object]:
    built = catalog.build_catalog(paths.square, paths.bin_csv)
    joined_paths = set(built.frame["path"])
    split_indices, split_summary = splits_section(built.frame, seed)

    report: dict[str, object] = {
        "catalog": catalog_section(built),
        "geometry": geometry_section(paths.square, joined_paths),
        "padding": padding_section(paths.square, paths.original, padding_samples),
        "families": families_section(built, paths.raw_csv),
        "colorways": colorways_section(built.frame),
        "directory_agreement": directory_section(built),
        "splits": split_summary,
        "labels": labels_section(built, split_indices, threshold),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarise(report: dict) -> str:
    cat = report["catalog"]
    pad = report["padding"]
    geo = report["geometry"]
    col = report["colorways"]
    dirs = report["directory_agreement"]
    spl = report["splits"]["counts"]
    lab = report["labels"]

    lines: list[str] = [
        f"catalog    files {cat['image_files']}  metadata {cat['metadata_rows']}  "
        f"dups {cat['duplicate_cids']}  orphans {cat['orphan_files']}  joined {cat['joined_rows']}",
    ]
    if pad.get("available"):
        lines.append(
            f"padding    {pad['verdict']}  top offset {pad['offset_top']} over {pad['pairs']} pairs  "
            f"residual {pad['residual_mae_median']}  cropped {pad['cropped_cases']}"
        )
    else:
        lines.append(f"padding    skipped ({pad['reason']})")
    lines.append(
        f"geometry   non-conforming {geo['non_conforming_total']}, "
        f"surviving join {geo['non_conforming_in_join']}"
    )

    header = f"{'family':13s}{'kind':9s}{'lbls':>5s}{'observed':>10s}{'multi':>8s}{'%obs':>7s}{'max':>5s}"
    lines += ["", header + "  xcheck"]
    for row in report["families"]["families"]:
        lines.append(
            f"{row['name']:13s}{row['kind']:9s}{row['labels']:>5d}{row['observed']:>10d}"
            f"{row['multi_positive']:>8d}{row['pct_of_observed']:>7.2f}{row['max_positives']:>5d}"
            f"  {'ok' if row['cross_check_ok'] else 'MISMATCH'}"
        )

    lines += [
        "",
        f"colorways  products {col['products']}  single {col['single_colorway_products']} "
        f"({col['single_colorway_pct']}%)  usable {col['usable_query_images']}",
        f"directory  Category {dirs['Category']['agree']}/{dirs['rows']}  "
        f"SubCategory {dirs['SubCategory']['agree']}/{dirs['rows']}",
        f"splits     train {spl['train']}  val {spl['val']}  test {spl['test']}  leakage passed",
        f"labels     threshold {lab['threshold']}  surviving {lab['surviving']}/{lab['total_labels']}  "
        f"softmax {lab['softmax_labels']}  bce {lab['bce_labels']}",
        f"           uncalibrated {len(lab['uncalibrated_labels'])}  "
        f"zero-train {len(lab['zero_train_positives'])}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure dataset facts and write artifacts/eda.json")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eda.json"))
    parser.add_argument("--threshold", type=int, default=catalog.MIN_LABEL_POSITIVES)
    parser.add_argument("--seed", type=int, default=splits.DEFAULT_SEED)
    parser.add_argument("--padding-samples", type=int, default=250)
    args = parser.parse_args()

    paths = DataPaths.from_root(args.data_root)
    report = run(paths, args.out, args.threshold, args.seed, args.padding_samples)
    print(summarise(report))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
