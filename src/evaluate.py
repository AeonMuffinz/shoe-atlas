"""Scores one trained run on validation: calibrate, threshold, then measure. Test stays locked."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import data_setup, metrics, model_builder, reporting
from src.catalog import FAMILIES, LabelSchema
from src.reporting import (
    CALIBRATION_NAME,
    ERRORS_DIR,
    PROBS_NAME,
    PROBS_OOF_NAME,
    PROCESSED_DIR,
    TEST_WITHHELD,
    THRESHOLDS_NAME,
    EvaluationError,
    Predictions,
)

WORST_ERRORS: int = 16


def load_checkpoint(run_dir: Path, name: str = "best.pt") -> dict:
    path = run_dir / name
    if not path.exists():
        raise EvaluationError(f"{path} not found; train the run before evaluating it")
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_schema_matches(payload: dict, schema: LabelSchema) -> None:
    stored = [str(c) for c in payload.get("schema_columns", [])]
    if stored != list(schema.columns):
        raise EvaluationError(
            f"checkpoint was trained on {len(stored)} labels and data/processed declares "
            f"{len(schema.columns)}; scoring them against each other would be meaningless"
        )


def build_from_checkpoint(payload: dict, num_labels: int, device: torch.device) -> nn.Module:
    config = dict(payload["config"])
    cfg = model_builder.ModelConfig.from_dict({**config, "num_labels": num_labels, "pretrained": False})
    model = model_builder.build_model(cfg)
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model: nn.Module, loader: object, device: torch.device) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    for images, _, _ in loader:  # type: ignore[union-attr]
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            chunks.append(model(images).float().cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def gather(artifacts: data_setup.Artifacts, split: str, logits: np.ndarray) -> Predictions:
    rows = artifacts.splits[split]
    return Predictions(
        logits=logits,
        labels=artifacts.labels[rows].astype(np.float64),
        family_observed=artifacts.family_observed[rows],
        rows=rows,
    )


def worst_error_rows(
    calibrated: np.ndarray, predictions: Predictions, schema: LabelSchema, family: str, count: int
) -> np.ndarray:
    slice_ = schema.family(family)
    observed = np.flatnonzero(predictions.family_observed[:, FAMILIES.index(family)])
    if observed.size == 0:
        return np.empty(0, dtype=np.int64)
    block = calibrated[observed, slice_.start : slice_.end]
    truth = predictions.labels[observed, slice_.start : slice_.end].argmax(axis=1)
    predicted = block.argmax(axis=1)
    wrong = predicted != truth
    if not wrong.any():
        return np.empty(0, dtype=np.int64)
    confidence = block.max(axis=1)
    ranked = observed[wrong][np.argsort(-confidence[wrong])]
    return ranked[:count]


def write_error_montages(
    run_dir: Path,
    calibrated: np.ndarray,
    predictions: Predictions,
    schema: LabelSchema,
    processed_dir: Path,
    count: int = WORST_ERRORS,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = run_dir / ERRORS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    images = np.load(processed_dir / data_setup.IMAGES_NAME, mmap_mode="r")

    for family in (f.name for f in schema.softmax_families()):
        picked = worst_error_rows(calibrated, predictions, schema, family, count)
        if picked.size == 0:
            continue
        slice_ = schema.family(family)
        columns = min(4, len(picked))
        rows = int(np.ceil(len(picked) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows))
        for axis in np.atleast_1d(axes).ravel():
            axis.axis("off")
        for axis, position in zip(np.atleast_1d(axes).ravel(), picked, strict=False):
            catalog_row = int(predictions.rows[position])
            block = calibrated[position, slice_.start : slice_.end]
            truth = slice_.labels[int(predictions.labels[position, slice_.start : slice_.end].argmax())]
            guess = slice_.labels[int(block.argmax())]
            axis.imshow(np.asarray(images[catalog_row]))
            axis.set_title(f"true {truth}\npred {guess} ({block.max():.2f})", fontsize=7)
        figure.suptitle(f"{family}: most confident errors", fontsize=10)
        figure.tight_layout()
        figure.savefig(folder / f"{family}.png", dpi=110)
        plt.close(figure)


def evaluate_run(
    run_dir: Path,
    processed_dir: Path = PROCESSED_DIR,
    checkpoint: str = "best.pt",
    unlock_test: bool = False,
    device: torch.device | None = None,
    montages: bool = True,
) -> dict[str, object]:
    artifacts = data_setup.load_artifacts(processed_dir)
    schema = artifacts.schema
    payload = load_checkpoint(run_dir, checkpoint)
    assert_schema_matches(payload, schema)

    run_name = run_dir.name
    winner = reporting.assert_test_unlocked(run_name) if unlock_test else None

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_from_checkpoint(payload, len(schema.columns), device)
    config = dict(payload["config"])
    data_cfg = data_setup.DataConfig(
        processed_dir=processed_dir,
        image_size=int(config.get("image_size", 224)),
        batch_size=int(config.get("batch_size", 64)),
        num_workers=int(config.get("num_workers", 4)),
        seed=int(config.get("seed", 42)),
        crop_ratio=tuple(config.get("crop_ratio", data_setup.DEFAULT_CROP_RATIO)),
        mean=tuple(config.get("mean", data_setup.IMAGENET_MEAN)),
        std=tuple(config.get("std", data_setup.IMAGENET_STD)),
    )

    report: dict[str, object] = {
        "run": run_name,
        "stem": config.get("stem", run_name),
        "role": config.get("role", "candidate"),
        "checkpoint": checkpoint,
        "epoch": payload.get("epoch"),
        "selection_metric": config.get("monitor"),
    }

    splits_to_score = ["val"] + (["test"] if unlock_test else [])
    for split in splits_to_score:
        loader = data_setup.build_dataloader(data_cfg, artifacts, split)
        predictions = gather(artifacts, split, predict(model, loader, device))
        scores, calibrated, thresholds, calibrators = reporting.score_split(predictions, schema)
        reporting.assert_contract(scores, f"{run_name}/{split}")
        report[split] = scores
        np.save(run_dir / PROBS_NAME.format(split=split), calibrated.astype(np.float32))
        held_out, _ = metrics.out_of_fold_probabilities(
            predictions.logits, predictions.labels, predictions.family_observed, schema
        )
        np.save(run_dir / PROBS_OOF_NAME.format(split=split), held_out.astype(np.float32))
        if split == "val":
            reporting.write_confusion(run_dir, calibrated, predictions, schema)
            (run_dir / THRESHOLDS_NAME).write_text(
                json.dumps(
                    {name: float(thresholds[i]) for i, name in enumerate(schema.columns)}, indent=2
                ),
                encoding="utf-8",
            )
            (run_dir / CALIBRATION_NAME).write_text(
                json.dumps(calibrators.to_dict(), indent=2), encoding="utf-8"
            )
            if montages:
                write_error_montages(run_dir, calibrated, predictions, schema, processed_dir)

    if not unlock_test:
        report["test"] = TEST_WITHHELD
    else:
        report["test_unlocked_on"] = date.today().isoformat()
        report["winner_record"] = winner

    reporting.write_report(run_dir, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a trained run on validation")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--no-montages", action="store_true")
    parser.add_argument(
        "--unlock-test",
        action="store_true",
        help="score the test split; refuses unless artifacts/winner.json names this run",
    )
    args = parser.parse_args()

    report = evaluate_run(
        args.run,
        processed_dir=args.processed,
        checkpoint=args.checkpoint,
        unlock_test=args.unlock_test,
        montages=not args.no_montages,
    )
    val = report["val"]
    print(f"run            {report['run']}  (epoch {report['epoch']}, {report['checkpoint']})")
    print(f"mAP            {val['map']:.4f}   softmax {val['map_softmax']:.4f}   bce {val['map_bce']:.4f}")
    print(f"macro F1 bce   {val['macro_f1_bce']:.4f}   out-of-fold {val['macro_f1_bce_out_of_fold']:.4f}")
    print(
        f"calibration    {val['calibration_error']:.4f}   "
        f"uncalibrated {val['calibration_error_uncalibrated']:.4f}"
    )
    print(f"family top-1   {({k: round(v, 4) for k, v in val['family_top1'].items()})}")
    print(f"test           {report.get('test', 'evaluated')}")


if __name__ == "__main__":
    main()
