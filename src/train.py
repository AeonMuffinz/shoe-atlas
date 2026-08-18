"""Trains one backbone end to end: a frozen-backbone head warmup, then full fine-tuning with LLRD."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from timm.data import resolve_model_data_config
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src import data_setup, engine, losses, metrics, model_builder, utils
from src.catalog import LabelSchema

RUNS_ROOT: Path = Path("artifacts/runs")
PROCESSED_DIR: Path = Path("data/processed")
LAST_NAME: str = "last.pt"
BEST_NAME: str = "best.pt"
SUMMARY_NAME: str = "run_summary.json"
CONFIG_NAME: str = "config.yaml"
WARMUP_PHASE: str = "warmup"
FINETUNE_PHASE: str = "finetune"
SELECTION_SCOPE: str = "best epoch within this run; the architecture winner is chosen on mAP elsewhere"


class TrainingError(RuntimeError):
    pass


@dataclass
class BestState:
    value: float
    epoch: int
    mode: str

    def improves(self, candidate: float) -> bool:
        return candidate < self.value if self.mode == "min" else candidate > self.value


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    best: float = float("inf")
    waited: int = 0
    stopped_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "max" and self.best == float("inf"):
            self.best = float("-inf")

    def improved(self, candidate: float) -> bool:
        if self.mode == "min":
            return candidate < self.best - self.min_delta
        return candidate > self.best + self.min_delta

    def update(self, candidate: float, epoch: int) -> bool:
        if self.improved(candidate):
            self.best = candidate
            self.waited = 0
            return False
        self.waited += 1
        if self.waited >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


def assert_cuda(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise TrainingError(
        "CUDA is not available. Training on the CPU silently would waste hours; pass --cpu to override."
    )


def guard_existing_checkpoints(run_dir: Path, force: bool) -> None:
    existing = [run_dir / name for name in (BEST_NAME, LAST_NAME) if (run_dir / name).exists()]
    if not existing or force:
        return
    lines = [
        f"  {p.name}  {p.stat().st_size / 2**20:.0f} MiB  {time.ctime(p.stat().st_mtime)}"
        for p in existing
    ]
    raise TrainingError(
        "refusing to retrain over an existing checkpoint:\n"
        + "\n".join(lines)
        + "\npass --force to overwrite."
    )


def assert_artifacts_match_schema(artifacts: data_setup.Artifacts) -> None:
    manifest = artifacts.manifest
    if manifest and manifest.get("labels") != len(artifacts.schema.columns):
        raise TrainingError(
            f"manifest records {manifest.get('labels')} labels but the schema declares "
            f"{len(artifacts.schema.columns)}; data/processed is stale, rerun prepare_data"
        )
    if manifest and manifest.get("rows") != artifacts.labels.shape[0]:
        raise TrainingError("manifest row count disagrees with labels.npy; rerun prepare_data")


def build_scheduler(optimizer: Optimizer, epochs: int, steps_per_epoch: int, accum: int) -> LRScheduler:
    steps = max(1, epochs * max(1, steps_per_epoch // accum))
    return CosineAnnealingLR(optimizer, T_max=steps)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    schema: LabelSchema,
    config: dict,
    epoch: int,
    phase: str,
    best: BestState,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "schema_columns": list(schema.columns),
        "config": config,
        "epoch": epoch,
        "phase": phase,
        "best_value": best.value,
        "best_epoch": best.epoch,
        "monitor_mode": best.mode,
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_phase(
    model: nn.Module,
    loaders: dict,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    logger: utils.RunLogger,
    device: torch.device,
    config: dict,
    schema: LabelSchema,
    run_dir: Path,
    phase: str,
    epochs: range,
    best: BestState,
    stopper: EarlyStopping | None = None,
) -> tuple[BestState, bool]:
    monitor = str(config["monitor"])
    stopped = False
    for epoch in epochs:
        train_metrics = engine.train_one_epoch(
            model, loaders["train"], loss_fn, optimizer, scheduler, logger, device, epoch,
            accum_steps=int(config["grad_accum"]),
            max_grad_norm=float(config["max_grad_norm"]),
            log_every=int(config["log_every"]),
            phase=phase,
        )
        evaluation = engine.evaluate(
            model, loaders["val"], loss_fn, logger, device, epoch, phase="val",
            global_step=int(train_metrics["global_step"]),
        )
        val_metrics = dict(evaluation.metrics)
        val_metrics.update(validation_ranking(evaluation, schema))
        candidate = float(val_metrics[monitor])

        save_checkpoint(
            run_dir / LAST_NAME, model, schema, config, epoch, phase, best, optimizer, scheduler
        )
        if best.improves(candidate):
            best = BestState(value=candidate, epoch=epoch, mode=best.mode)
            save_checkpoint(run_dir / BEST_NAME, model, schema, config, epoch, phase, best)

        logger.log(
            {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(evaluation.metrics["loss"]),
                "val_loss_softmax": float(evaluation.metrics.get("loss_softmax", 0.0)),
                "val_loss_bce": float(evaluation.metrics.get("loss_bce", 0.0)),
                "val_map": float(val_metrics["map"]),
                "val_map_softmax": float(val_metrics["map_softmax"]),
                "val_map_bce": float(val_metrics["map_bce"]),
                "best_value": best.value,
            },
            step=int(train_metrics["global_step"]),
            phase="epoch",
        )
        print(
            f"[{phase}] epoch {epoch:2d}  train {train_metrics['loss']:.4f}  "
            f"val {evaluation.metrics['loss']:.4f}  mAP {val_metrics['map']:.4f}  "
            f"(soft {val_metrics['map_softmax']:.4f} / bce {val_metrics['map_bce']:.4f})  "
            f"best {best.value:.4f}@{best.epoch}"
        )
        if stopper is not None and stopper.update(candidate, epoch):
            print(
                f"[{phase}] early stop at epoch {epoch}: no {monitor} improvement for "
                f"{stopper.patience} epochs (best {best.value:.4f}@{best.epoch})"
            )
            stopped = True
            break
    return best, stopped


def validation_ranking(evaluation: engine.EvalOutputs, schema: LabelSchema) -> dict[str, float]:
    probs = metrics.logits_to_probabilities(evaluation.logits.numpy(), schema)
    labels = evaluation.targets.numpy()
    mask = metrics.cell_mask(evaluation.mask.numpy(), schema)
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


def run_name_for(stem: str, seed: int) -> str:
    return f"{stem}_s{seed}"


def train(config: dict, args: argparse.Namespace) -> dict:
    stem = str(config["name"])
    seed = int(config["seed"])
    name = run_name_for(stem, seed)
    config["stem"] = stem
    config["run_name"] = name
    run_dir = RUNS_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    guard_existing_checkpoints(run_dir, args.force)

    device = assert_cuda(args.cpu)
    utils.set_seed(int(config["seed"]))

    artifacts = data_setup.load_artifacts(PROCESSED_DIR)
    assert_artifacts_match_schema(artifacts)
    schema = artifacts.schema
    config["num_labels"] = len(schema.columns)

    model_cfg = model_builder.ModelConfig.from_dict({**config, "num_labels": len(schema.columns)})
    model = model_builder.build_model(model_cfg).to(device)
    data_cfg_source = resolve_model_data_config(model)

    data_cfg = data_setup.DataConfig(
        processed_dir=PROCESSED_DIR,
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=int(config["seed"]),
        crop_scale=tuple(config["crop_scale"]),
        mean=tuple(data_cfg_source["mean"]),
        std=tuple(data_cfg_source["std"]),
    )
    loaders = data_setup.build_dataloaders(data_cfg, artifacts)
    if args.limit_batches:
        loaders = {k: list(islice(v, args.limit_batches)) for k, v in loaders.items()}

    pos_weight = losses.compute_pos_weight(
        artifacts.labels, artifacts.family_observed, schema,
        artifacts.splits["train"], float(config["pos_weight_cap"]),
    )
    loss_fn = losses.FamilyLoss(schema, pos_weight, losses.LossConfig.from_dict(config)).to(device)

    config["steps_per_epoch"] = len(loaders["train"])
    config["mean"] = list(data_cfg.mean)
    config["std"] = list(data_cfg.std)
    (run_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    warmup_epochs = int(config["warmup_epochs"])
    total_epochs = int(config["max_epochs"])
    best = BestState(value=float("inf") if config["monitor_mode"] == "min" else float("-inf"),
                     epoch=-1, mode=str(config["monitor_mode"]))
    started = time.time()
    timings: dict[str, float] = {}
    params: dict[str, int] = {"total": model_builder.trainable_parameters(model)}

    with utils.make_logger(name, config, run_dir, use_wandb=not args.no_wandb, group=stem) as logger:
        frozen = model_builder.freeze_backbone(model)
        params["warmup_trainable"] = model_builder.trainable_parameters(model)
        logger.log(
            {"frozen_tensors": float(frozen), "trainable_params": float(params["warmup_trainable"]),
             "total_params": float(params["total"])},
            step=0, phase="phase_warmup_start",
        )
        print(f"warmup: {params['warmup_trainable']:,} of {params['total']:,} params trainable")

        phase_start = time.time()
        optimizer = model_builder.build_optimizer(model, model_cfg, model_builder.HEAD_PHASE)
        best, _ = run_phase(model, loaders, loss_fn, optimizer, None, logger, device, config, schema,
                            run_dir, WARMUP_PHASE, range(warmup_epochs), best)
        timings[WARMUP_PHASE] = time.time() - phase_start

        model_builder.unfreeze_all(model)
        params["finetune_trainable"] = model_builder.trainable_parameters(model)
        optimizer = model_builder.build_optimizer(model, model_cfg, model_builder.FULL_PHASE)
        distinct_lrs = sorted({round(float(g["lr"]), 12) for g in optimizer.param_groups})
        scheduler = build_scheduler(
            optimizer, total_epochs - warmup_epochs, len(loaders["train"]), int(config["grad_accum"])
        )
        logger.log(
            {
                "trainable_params": float(params["finetune_trainable"]),
                "distinct_lrs": float(len(distinct_lrs)),
                "lr_min": distinct_lrs[0],
                "lr_max": distinct_lrs[-1],
                "lr_head": distinct_lrs[-1],
            },
            step=warmup_epochs * len(loaders["train"]), phase="phase_finetune_start",
        )
        print(f"finetune: {params['finetune_trainable']:,} params trainable, "
              f"{len(distinct_lrs)} distinct lrs {distinct_lrs[0]:.2e}..{distinct_lrs[-1]:.2e}")

        phase_start = time.time()
        stopper = EarlyStopping(
            patience=int(config["patience"]),
            min_delta=float(config["min_delta"]),
            mode=str(config["monitor_mode"]),
        )
        best, stopped = run_phase(model, loaders, loss_fn, optimizer, scheduler, logger, device, config,
                                  schema, run_dir, FINETUNE_PHASE, range(warmup_epochs, total_epochs),
                                  best, stopper)
        timings[FINETUNE_PHASE] = time.time() - phase_start

        summary = {
            "name": name,
            "stem": stem,
            "selection_metric": f"val_{config['monitor']}",
            "selection_scope": SELECTION_SCOPE,
            "best_value": best.value,
            "best_epoch": best.epoch,
            "max_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "stopped_early": stopped,
            "stopped_at_epoch": stopper.stopped_epoch,
            "epochs_completed": (stopper.stopped_epoch + 1) if stopped else total_epochs,
            "patience": int(config["patience"]),
            "seed": int(config["seed"]),
            "device": str(device),
            "wall_clock_seconds": time.time() - started,
            "phase_seconds": timings,
            "params": params,
            "steps_per_epoch": len(loaders["train"]),
            "wandb_mode": "disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "online"),
            "torch_version": torch.__version__,
            "determinism": "seed-controlled, not bitwise deterministic",
        }
        logger.summary({"best_value": best.value, "best_epoch": float(best.epoch)})

    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one backbone through warmup then full fine-tuning")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
    parser.add_argument("--cpu", action="store_true", help="allow training without CUDA")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--limit-batches", type=int, default=0, help="smoke run with N batches per epoch"
    )
    args = parser.parse_args()

    config = utils.load_config(args.config)
    config.setdefault("name", args.config.stem)
    summary = train(config, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
