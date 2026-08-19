"""A linear head on frozen ImageNet features, answering whether fine-tuning earned its cost."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_model_data_config
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src import data_setup, engine, features, losses, reporting, selection, utils
from src.catalog import LabelSchema
from src.reporting import PROCESSED_DIR, RUNS_ROOT, TEST_WITHHELD
from src.selection import ConstrainedSelector, EarlyStopping

STEM: str = "linear_probe"
BACKBONE: str = "convnext_tiny.fb_in22k_ft_in1k"
FEATURE_ROOT: Path = Path("artifacts")
BEST_NAME: str = "best.pt"
PHASE: str = "probe"
SELECTION_SCOPE: str = "best epoch within this run; the architecture winner is chosen on mAP elsewhere"

NOTE_AUGMENTATION: str = (
    "features are extracted once from unaugmented images, so this run trains without the "
    "augmentation the fine-tuned runs receive. The deviation disadvantages the probe, which makes "
    "a competitive result stronger rather than weaker; the linear head is small enough that "
    "augmentation's regularising role is close to irrelevant."
)
NOTE_LAYER_DECAY: str = (
    "layer-wise decay is inapplicable rather than skipped: the trainable model is one linear layer, "
    "so there are no layers to decay and no group_matcher to consult"
)


class ProbeError(RuntimeError):
    pass


def run_name_for(seed: int) -> str:
    return f"{STEM}_s{seed}"


def cache_tag(backbone: str) -> str:
    return backbone.replace(".", "_").replace("/", "_")


def guard_existing_checkpoint(run_dir: Path, force: bool) -> None:
    path = run_dir / BEST_NAME
    if not path.exists() or force:
        return
    raise ProbeError(
        f"refusing to retrain over {path} ({path.stat().st_size / 2**10:.0f} KiB, "
        f"{time.ctime(path.stat().st_mtime)}); pass --force to overwrite."
    )


def build_encoder(backbone: str, device: torch.device) -> nn.Module:
    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    return model.to(device).eval()


def split_features(
    encoder: nn.Module,
    artifacts: data_setup.Artifacts,
    data_cfg: data_setup.DataConfig,
    split: str,
    device: torch.device,
    feature_root: Path,
    backbone: str,
) -> np.ndarray:
    path = features.cache_path(feature_root, cache_tag(backbone), data_cfg.image_size, split)
    rows = len(artifacts.splits[split])

    def extract() -> np.ndarray:
        loader = data_setup.build_eval_dataloader(data_cfg, artifacts, split)
        return features.extract_features(encoder, loader, device)

    return features.load_or_extract(path, extract, rows=rows)


def tensor_loader(
    matrix: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(matrix.astype(np.float32)),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(family_observed.astype(np.bool_)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=0,
        generator=generator,
    )


def save_head(path: Path, head: nn.Module, schema: LabelSchema, config: dict, epoch: int) -> None:
    payload = {
        "model": head.state_dict(),
        "schema_columns": list(schema.columns),
        "config": config,
        "epoch": epoch,
        "phase": PHASE,
        "torch_version": torch.__version__,
    }
    torch.save(payload, path)


def fit_head(
    head: nn.Module,
    loaders: dict[str, DataLoader],
    loss_fn: nn.Module,
    optimizer: AdamW,
    scheduler: object,
    logger: utils.RunLogger,
    device: torch.device,
    schema: LabelSchema,
    config: dict,
    run_dir: Path,
    max_epochs: int,
    selector: ConstrainedSelector,
    stopper: EarlyStopping,
    history: list[tuple[int, dict[str, float]]],
) -> bool:
    stopped = False
    for epoch in range(max_epochs):
        train_metrics = engine.train_one_epoch(
            head, loaders["train"], loss_fn, optimizer, scheduler, logger, device, epoch,
            max_grad_norm=float(config["max_grad_norm"]),
            log_every=int(config["log_every"]),
            use_amp=False,
            phase=PHASE,
        )
        evaluation = engine.evaluate(
            head, loaders["val"], loss_fn, logger, device, epoch, phase="val",
            use_amp=False, global_step=int(train_metrics["global_step"]),
        )
        val_metrics = dict(evaluation.metrics)
        val_metrics.update(
            reporting.ranking_from_logits(
                evaluation.logits.numpy(), evaluation.targets.numpy(), evaluation.mask.numpy(), schema
            )
        )
        accepted = selector.update(val_metrics, epoch)
        history.append((epoch, dict(val_metrics)))
        if accepted:
            save_head(run_dir / BEST_NAME, head, schema, config, epoch)

        logger.log(
            {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(evaluation.metrics["loss"]),
                "val_map": float(val_metrics["map"]),
                "val_map_softmax": float(val_metrics["map_softmax"]),
                "val_map_bce": float(val_metrics["map_bce"]),
                "best_value": selector.value,
            },
            step=int(train_metrics["global_step"]),
            phase="epoch",
        )
        print(
            f"[{PHASE}] epoch {epoch:2d}  train {train_metrics['loss']:.4f}  "
            f"val {evaluation.metrics['loss']:.4f}  mAP {val_metrics['map']:.4f}  "
            f"(soft {val_metrics['map_softmax']:.4f} / bce {val_metrics['map_bce']:.4f})  "
            f"best {selector.value:.4f}@{selector.epoch}"
        )
        if stopper.update(float(val_metrics[selector.progress_metric()]), epoch):
            print(
                f"[{PHASE}] early stop at epoch {epoch}: {selector.progress_metric()} did not improve "
                f"for {stopper.patience} epochs"
            )
            stopped = True
            break
    return stopped


def default_config(seed: int, image_size: int) -> dict:
    return {
        "name": STEM,
        "stem": STEM,
        "backbone": BACKBONE,
        "seed": seed,
        "image_size": image_size,
        "batch_size": 512,
        "extract_batch_size": 128,
        "num_workers": 4,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "max_epochs": 60,
        "patience": 4,
        "min_delta": 0.0,
        "max_grad_norm": 1.0,
        "log_every": 20,
        "pos_weight_cap": 10.0,
        "softmax_weight": 1.0,
        "bce_weight": 1.0,
        "role": "candidate",
        "monitor": "constrained",
        "monitor_mode": "max",
        "selection_primary": "map_bce",
        "selection_guard": "map_softmax",
        "selection_epsilon": 0.01,
    }


def run(
    config: dict,
    processed_dir: Path = PROCESSED_DIR,
    runs_root: Path = RUNS_ROOT,
    feature_root: Path = FEATURE_ROOT,
    device: torch.device | None = None,
    force: bool = False,
    use_wandb: bool = True,
) -> dict[str, object]:
    seed = int(config["seed"])
    name = run_name_for(seed)
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    guard_existing_checkpoint(run_dir, force)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(seed)

    artifacts = data_setup.load_artifacts(processed_dir)
    schema = artifacts.schema
    encoder = build_encoder(str(config["backbone"]), device)
    source = resolve_model_data_config(encoder)

    data_cfg = data_setup.DataConfig(
        processed_dir=processed_dir,
        image_size=int(config["image_size"]),
        batch_size=int(config["extract_batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=seed,
        mean=tuple(source["mean"]),
        std=tuple(source["std"]),
    )

    started = time.time()
    cached = {
        split: split_features(
            encoder, artifacts, data_cfg, split, device, feature_root, str(config["backbone"])
        )
        for split in ("train", "val")
    }
    extraction_seconds = time.time() - started
    dimension = int(cached["train"].shape[1])

    loaders = {
        split: tensor_loader(
            cached[split],
            artifacts.labels[artifacts.splits[split]],
            artifacts.family_observed[artifacts.splits[split]],
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            seed=seed,
        )
        for split in ("train", "val")
    }

    head = nn.Linear(dimension, len(schema.columns)).to(device)
    pos_weight = losses.compute_pos_weight(
        artifacts.labels, artifacts.family_observed, schema,
        artifacts.splits["train"], float(config["pos_weight_cap"]),
    )
    loss_fn = losses.FamilyLoss(schema, pos_weight, losses.LossConfig.from_dict(config)).to(device)
    optimizer = AdamW(
        head.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["max_epochs"]) * len(loaders["train"]))
    )
    selector = ConstrainedSelector(
        primary=str(config["selection_primary"]),
        guard=str(config["selection_guard"]),
        epsilon=float(config["selection_epsilon"]),
    )
    stopper = EarlyStopping(
        patience=int(config["patience"]),
        min_delta=float(config["min_delta"]),
        mode=str(config["monitor_mode"]),
    )

    config["feature_dim"] = dimension
    config["mean"] = list(data_cfg.mean)
    config["std"] = list(data_cfg.std)
    config["run_name"] = name
    config["steps_per_epoch"] = len(loaders["train"])
    (run_dir / reporting.CONFIG_NAME).write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    with utils.make_logger(name, config, run_dir, use_wandb=use_wandb, group=STEM) as logger:
        print(f"probe: {trainable:,} trainable params on {dimension}-d frozen features")
        history: list[tuple[int, dict[str, float]]] = []
        stopped = fit_head(
            head, loaders, loss_fn, optimizer, scheduler, logger, device, schema, config,
            run_dir, int(config["max_epochs"]), selector, stopper, history,
        )
        guard_audit = selection.assert_online_matches_offline(selector, history)
        logger.summary({"best_value": selector.value, "best_epoch": float(selector.epoch)})

    payload = torch.load(run_dir / BEST_NAME, map_location=device, weights_only=False)
    head.load_state_dict(payload["model"])
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(cached["val"].astype(np.float32)).to(device))
    rows = artifacts.splits["val"]
    predictions = reporting.Predictions(
        logits=logits.float().cpu().numpy().astype(np.float64),
        labels=artifacts.labels[rows].astype(np.float64),
        family_observed=artifacts.family_observed[rows],
        rows=rows,
    )
    scores, calibrated, thresholds, calibrators = reporting.score_split(predictions, schema)
    reporting.assert_contract(scores, name)

    report: dict[str, object] = {
        "run": name,
        "stem": STEM,
        "role": "candidate",
        "checkpoint": BEST_NAME,
        "epoch": selector.epoch,
        "selection_metric": "constrained",
        "backbone": str(config["backbone"]),
        "feature_dim": dimension,
        "trainable_params": trainable,
        "note_augmentation": NOTE_AUGMENTATION,
        "note_layer_decay": NOTE_LAYER_DECAY,
        "val": scores,
        "test": TEST_WITHHELD,
    }
    reporting.write_report(run_dir, report)
    reporting.write_confusion(run_dir, calibrated, predictions, schema)
    np.save(run_dir / reporting.PROBS_NAME.format(split="val"), calibrated.astype(np.float32))
    (run_dir / reporting.THRESHOLDS_NAME).write_text(
        json.dumps({name_: float(thresholds[i]) for i, name_ in enumerate(schema.columns)}, indent=2),
        encoding="utf-8",
    )
    (run_dir / reporting.CALIBRATION_NAME).write_text(
        json.dumps(calibrators.to_dict(), indent=2), encoding="utf-8"
    )
    reporting.write_summary(
        run_dir,
        {
            "name": name,
            "stem": STEM,
            "role": "candidate",
            **selector.describe(),
            "selection_scope": SELECTION_SCOPE,
            "best_value": selector.value,
            "best_epoch": selector.epoch,
            "max_epochs": int(config["max_epochs"]),
            "warmup_epochs": 0,
            "stopped_early": stopped,
            "stopped_at_ceiling": not stopped,
            "stop_reason": ("primary_plateau" if stopped else "max_epochs_ceiling"),
            "progress_metric": f"val_{selector.progress_metric()}",
            **guard_audit,
            "stopped_at_epoch": stopper.stopped_epoch,
            "epochs_completed": (stopper.stopped_epoch + 1) if stopped else int(config["max_epochs"]),
            "patience": int(config["patience"]),
            "seed": seed,
            "device": str(device),
            "wall_clock_seconds": time.time() - started,
            "phase_seconds": {"feature_extraction": extraction_seconds},
            "params": {"total": trainable, "trainable": trainable},
            "steps_per_epoch": len(loaders["train"]),
            "backbone": str(config["backbone"]),
            "feature_dim": dimension,
            "note_augmentation": NOTE_AUGMENTATION,
            "torch_version": torch.__version__,
            "determinism": "seed-controlled, not bitwise deterministic",
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear head on frozen ImageNet features")
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = default_config(args.seed, args.image_size)
    config["backbone"] = args.backbone
    if args.lr is not None:
        config["lr"] = args.lr
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs

    report = run(
        config,
        processed_dir=args.processed,
        runs_root=args.runs_root,
        feature_root=args.feature_root,
        device=torch.device("cpu") if args.cpu else None,
        force=args.force,
        use_wandb=not args.no_wandb,
    )
    val = report["val"]
    print(f"run            {report['run']}  (epoch {report['epoch']}, {report['feature_dim']}-d features)")
    print(f"mAP            {val['map']:.4f}   softmax {val['map_softmax']:.4f}   bce {val['map_bce']:.4f}")
    print(f"macro F1 bce   {val['macro_f1_bce']:.4f}   out-of-fold {val['macro_f1_bce_out_of_fold']:.4f}")
    print(
        f"calibration    {val['calibration_error']:.4f}   "
        f"uncalibrated {val['calibration_error_uncalibrated']:.4f}"
    )
    print(f"family top-1   {({k: round(v, 4) for k, v in val['family_top1'].items()})}")


if __name__ == "__main__":
    main()
