"""Per-label error analysis: is average precision explained by label frequency, or by the model?

Runs against a completed run's persisted validation probabilities. No GPU, no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import metrics
from src.catalog import FAMILIES, LabelSchema

PROCESSED = Path("data/processed")
RUNS = Path("artifacts/runs")
EDA = Path("artifacts/eda.json")
NOISE_FLOOR = 0.001
STRATA: tuple[tuple[str, int, int], ...] = (
    ("common (>1000)", 1001, 10**9),
    ("mid (200-1000)", 200, 1000),
    ("rare (50-200)", 0, 199),
)
HEEL_ORDER: tuple[str, ...] = (
    "HeelHeight.Flat",
    "HeelHeight.Under.1in",
    "HeelHeight.1in...1.3.4in",
    "HeelHeight.2in...2.3.4in",
    "HeelHeight.3in...3.3.4in",
    "HeelHeight.4in...4.3.4in",
    "HeelHeight.5in...over",
)
KNOWN_CEILINGS: dict[str, str] = {
    "ToeStyle.Moc Toe": "near-identical to Algonquin in a photograph",
    "ToeStyle.Algonquin": "near-identical to Moc Toe in a photograph",
    "SubCategory.Prewalker": "differs from Firstwalker by sole stiffness only",
    "SubCategory.Firstwalker": "differs from Prewalker by sole stiffness only",
    "Material.Terry": "91% concentrated in Slipper Flats, a 35x concentration",
}


def per_label_ap(
    probs: np.ndarray, labels: np.ndarray, family_observed: np.ndarray, schema: LabelSchema
) -> dict[str, float]:
    mask = metrics.cell_mask(family_observed, schema)
    values = metrics.per_label_average_precision(probs, labels, mask)
    return {name: float(values[i]) for i, name in enumerate(schema.columns)}


def stratum_of(count: int) -> str:
    for name, low, high in STRATA:
        if low <= count <= high:
            return name
    return STRATA[-1][0]


def group_of(name: str, schema: LabelSchema) -> str:
    family = schema.family(name.split(".", 1)[0])
    return "exclusive" if family.kind == "softmax" else "bce"


def stratified_map(
    ap: dict[str, float], counts: dict[str, int], schema: LabelSchema
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for stratum, _low, _high in STRATA:
        rows = {n: v for n, v in ap.items() if stratum_of(counts[n]) == stratum and np.isfinite(v)}
        entry: dict[str, float] = {"labels": float(len(rows))}
        if rows:
            entry["map"] = float(np.mean(list(rows.values())))
            for group in ("exclusive", "bce"):
                subset = [v for n, v in rows.items() if group_of(n, schema) == group]
                entry[f"map_{group}"] = float(np.mean(subset)) if subset else float("nan")
                entry[f"n_{group}"] = float(len(subset))
        out[stratum] = entry
    return out


def adjacency_profile(matrix: np.ndarray, names: list[str], order: tuple[str, ...]) -> dict[str, float]:
    rank = {name: i for i, name in enumerate(order)}
    if not set(names) <= set(rank):
        raise ValueError(f"labels missing from the ordering: {sorted(set(names) - set(rank))}")
    distances: dict[int, int] = {}
    errors = 0
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i == j:
                continue
            count = int(matrix[i, j])
            if not count:
                continue
            step = abs(rank[true_name] - rank[pred_name])
            distances[step] = distances.get(step, 0) + count
            errors += count
    profile = {f"distance_{k}": distances.get(k, 0) / errors for k in sorted(distances)}
    profile["errors"] = float(errors)
    profile["adjacent_fraction"] = distances.get(1, 0) / errors if errors else float("nan")
    return profile


def brand_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
    brands: np.ndarray,
    family: str,
    minimum: int = 30,
) -> pd.DataFrame:
    slice_ = schema.family(family)
    observed = np.flatnonzero(family_observed[:, FAMILIES.index(family)])
    truth = labels[observed, slice_.start : slice_.end].argmax(axis=1)
    predicted = probs[observed, slice_.start : slice_.end].argmax(axis=1)
    frame = pd.DataFrame(
        {"brand": brands[observed], "correct": (truth == predicted).astype(float)}
    )
    grouped = frame.groupby("brand").agg(rows=("correct", "size"), accuracy=("correct", "mean"))
    return grouped[grouped["rows"] >= minimum].sort_values("accuracy")


def scatter(ap: dict[str, float], counts: dict[str, int], schema: LabelSchema, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for group, colour in (("exclusive", "#c2410c"), ("bce", "#1d4ed8")):
        xs = [counts[n] for n, v in ap.items() if np.isfinite(v) and group_of(n, schema) == group]
        ys = [v for n, v in ap.items() if np.isfinite(v) and group_of(n, schema) == group]
        axis.scatter(xs, ys, s=22, alpha=0.75, c=colour, label=f"{group} ({len(xs)})")
    axis.set_xscale("log")
    axis.set_xlabel("training positives (log scale)")
    axis.set_ylabel("average precision, validation")
    axis.set_title("Per-label AP against label frequency")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-label error analysis for a completed run")
    parser.add_argument("--run", type=str, default="convnext_tiny_s42")
    parser.add_argument("--bottom", type=int, default=15)
    args = parser.parse_args()

    run_dir = RUNS / args.run
    schema = LabelSchema.load(PROCESSED / "label_schema.json")
    labels_all = np.load(PROCESSED / "labels.npy")
    observed_all = np.load(PROCESSED / "family_observed.npy")
    splits = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))["indices"]
    val = np.asarray(splits["val"], dtype=np.int64)
    probs = np.load(run_dir / "val_probs.npy").astype(np.float64)
    labels = labels_all[val].astype(np.float64)
    observed = observed_all[val]

    eda = json.loads(EDA.read_text(encoding="utf-8"))["labels"]
    train_counts = {k: int(v) for k, v in eda["train_positives"].items()}
    val_counts = {k: int(v) for k, v in eda["val_positives"].items()}

    ap = per_label_ap(probs, labels, observed, schema)
    finite = {n: v for n, v in ap.items() if np.isfinite(v)}

    print(f"run {args.run}: per-label AP on CALIBRATED probabilities "
          f"({len(finite)} of {len(ap)} scoreable)")
    print("basis note: the headline mAP is uncalibrated; FINDINGS 6.2 measures the shift as under")
    print("0.008 per label, systematic but small. Recomputing uncalibrated needs a forward pass.\n")

    xs = np.array([train_counts[n] for n in finite])
    ys = np.array([finite[n] for n in finite])
    rho, p = spearmanr(xs, ys)
    print(f"1a  Spearman rank correlation of AP against training positives: rho={rho:.3f}, p={p:.2e}")
    for group in ("exclusive", "bce"):
        names = [n for n in finite if group_of(n, schema) == group]
        r, pv = spearmanr([train_counts[n] for n in names], [finite[n] for n in names])
        print(f"      {group:<10} rho={r:.3f}, p={pv:.2e}, n={len(names)}")
    scatter(ap, train_counts, schema, Path("artifacts/per_label_ap.png"))
    print("      scatter written to artifacts/per_label_ap.png")

    print("\n1b  mAP by frequency stratum")
    strata = stratified_map(ap, train_counts, schema)
    print(f"      {'stratum':<18}{'labels':>7}{'mAP':>9}{'exclusive':>11}{'bce':>9}")
    for name, row in strata.items():
        print(f"      {name:<18}{row['labels']:>7.0f}{row.get('map', float('nan')):>9.4f}"
              f"{row.get('map_exclusive', float('nan')):>11.4f}{row.get('map_bce', float('nan')):>9.4f}")

    print(f"\n1c  bottom {args.bottom} labels by AP")
    worst = sorted(finite.items(), key=lambda kv: kv[1])[: args.bottom]
    known = 0
    print(f"      {'label':<34}{'AP':>8}{'train':>8}{'val':>7}  known ceiling")
    for name, value in worst:
        note = KNOWN_CEILINGS.get(name, "")
        if name.startswith("Insole."):
            note = note or "Insole family, excluded from zero-shot as not prompt-distinguishable"
        if note:
            known += 1
        print(f"      {name:<34}{value:>8.4f}{train_counts[name]:>8}{val_counts[name]:>7}  {note}")
    print(f"      {known} of {len(worst)} were already recorded as visually ambiguous or excluded")

    print("\n1d  confusion, exclusive families")
    for family in schema.softmax_families():
        path = run_dir / "confusion" / f"{family.name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0)
        matrix = frame.to_numpy()
        total = matrix.sum()
        correct = np.trace(matrix)
        print(f"      {family.name}: {correct}/{total} correct ({correct/total:.4f})")
        if family.name == "HeelHeight":
            profile = adjacency_profile(matrix, list(frame.index), HEEL_ORDER)
            print(f"        errors {profile['errors']:.0f}, "
                  f"adjacent (one step) {profile['adjacent_fraction']:.4f}")
            for key in sorted(k for k in profile if k.startswith("distance_")):
                print(f"        {key}: {profile[key]:.4f}")

    print("\n1e  brand stratification, Category top-1")
    catalog = pd.read_csv(PROCESSED / "catalog.csv")
    brands = catalog["brand"].to_numpy()[val]
    table = brand_accuracy(probs, labels, observed, schema, brands, "Category")
    covered = int(table["rows"].sum())
    print("      594 distinct brands in validation; the largest has 181 rows, so no brand reaches")
    print(f"      200. Using a 30-row floor: {len(table)} brands covering {covered} of {len(val)} rows")
    print(f"      worst 5:\n{table.head(5).to_string()}")
    print(f"      best 5:\n{table.tail(5).to_string()}")
    print(f"      spread {table['accuracy'].max() - table['accuracy'].min():.4f}")


if __name__ == "__main__":
    main()
