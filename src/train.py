"""Trains one backbone end to end: a frozen-backbone head warmup, then full fine-tuning with LLRD."""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path

import torch
from timm.data import resolve_model_data_config
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src import data_setup, engine, losses, model_builder, reporting, selection, utils
from src.catalog import LabelSchema
from src.reporting import CONFIG_NAME, PROCESSED_DIR, RUNS_ROOT
from src.selection import BestState, ConstrainedSelector, EarlyStopping, ScalarSelector, make_selector

LAST_NAME: str = "last.pt"
BEST_NAME: str = "best.pt"
ELIGIBLE_NAME: str = "eligible_ep{epoch:03d}.pt"
WARMUP_PHASE: str = "warmup"
FINETUNE_PHASE: str = "finetune"
SELECTION_SCOPE: str = "best epoch within this run; the architecture winner is chosen on mAP elsewhere"


class TrainingError(RuntimeError):
    pass


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
    selector: ScalarSelector | ConstrainedSelector,
    stopper: EarlyStopping | selection.SelectionComplete | None = None,
    history: list[tuple[int, dict[str, float]]] | None = None,
    eligible: list[int] | None = None,
) -> bool:
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
        accepted = selector.update(val_metrics, epoch)
        best = selector.state()
        if history is not None:
            history.append((epoch, dict(val_metrics)))

        save_checkpoint(
            run_dir / LAST_NAME, model, schema, config, epoch, phase, best, optimizer, scheduler
        )
        if accepted:
            save_checkpoint(run_dir / BEST_NAME, model, schema, config, epoch, phase, best)
        if eligible is not None and getattr(selector, "last_guard_ok", False):
            eligible.append(epoch)
            save_checkpoint(
                run_dir / ELIGIBLE_NAME.format(epoch=epoch), model, schema, config, epoch, phase, best
            )

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
        if stopper is not None:
            fired = (
                stopper.update(val_metrics, epoch)
                if isinstance(stopper, selection.SelectionComplete)
                else stopper.update(float(val_metrics[selector.progress_metric()]), epoch)
            )
            if fired:
                print(
                    f"[{phase}] early stop at epoch {epoch}: {selector.progress_metric()} did not improve "
                    f"for {stopper.patience} epochs (best {best.value:.4f}@{best.epoch})"
                )
                stopped = True
                break
    return stopped


def validation_ranking(evaluation: engine.EvalOutputs, schema: LabelSchema) -> dict[str, float]:
    return reporting.ranking_from_logits(
        evaluation.logits.numpy(), evaluation.targets.numpy(), evaluation.mask.numpy(), schema
    )


CANDIDATE: str = "candidate"
PROBE: str = "probe"


def probe_metadata(config: dict) -> dict[str, object]:
    role = str(config.get("role", CANDIDATE))
    if role != PROBE:
        return {"role": role}
    varies = config.get("probe_varies")
    reference = config.get("probe_reference_run")
    if not varies or not reference:
        raise TrainingError(
            "a run with role: probe must declare probe_varies and probe_reference_run, so a reader "
            "can tell what it varies and against what without diffing configs"
        )
    if varies not in config:
        raise TrainingError(f"probe_varies is {varies!r} but that key is not in the config")
    return {
        "role": role,
        "probe": {
            "varies": str(varies),
            "value": config[varies],
            "reference_value": config.get("probe_reference_value"),
            "reference_run": str(reference),
        },
    }


def rotate_metrics(run_dir: Path) -> Path | None:
    path = run_dir / utils.METRICS_FILENAME
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(path.stat().st_mtime))
    archived = run_dir / f"metrics_{stamp}.jsonl.bak"
    os.replace(path, archived)
    return archived


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

    archived = rotate_metrics(run_dir)
    if archived is not None:
        print(f"archived previous metrics to {archived.name}")

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
    selector = make_selector(config)
    history: list[tuple[int, dict[str, float]]] = []
    eligible: list[int] = []
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
        run_phase(model, loaders, loss_fn, optimizer, None, logger, device, config, schema,
                  run_dir, WARMUP_PHASE, range(warmup_epochs), selector, history=history, eligible=eligible)
        timings[WARMUP_PHASE] = time.time() - phase_start

        model_builder.unfreeze_all(model)
        params["finetune_trainable"] = model_builder.trainable_parameters(model)
        optimizer = model_builder.build_optimizer(model, model_cfg, model_builder.FULL_PHASE)
        distinct_lrs = sorted({round(float(g["lr"]), 12) for g in optimizer.param_groups})
        if config.get("expected_backbone_floor") is not None:
            model_builder.assert_backbone_floor(
                optimizer, float(config["expected_backbone_floor"])
            )
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
        stopper = selection.make_stopper(config, selector)
        stopped = run_phase(model, loaders, loss_fn, optimizer, scheduler, logger, device, config,
                            schema, run_dir, FINETUNE_PHASE, range(warmup_epochs, total_epochs),
                            selector, stopper, history=history, eligible=eligible)
        timings[FINETUNE_PHASE] = time.time() - phase_start

        guard_audit: dict[str, object] = {}
        if isinstance(selector, ConstrainedSelector):
            guard_audit = selection.assert_online_matches_offline(selector, history)
            print(f"guard audit: online and offline both select epoch "
                  f"{guard_audit['offline_selected_epoch']} "
                  f"({guard_audit['offline_eligible_epochs']} of {len(history)} epochs eligible)")

        summary = {
            "name": name,
            "stem": stem,
            **probe_metadata(config),
            **selector.describe(),
            "selection_scope": SELECTION_SCOPE,
            "best_value": selector.value,
            "best_epoch": selector.epoch,
            "max_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "stopping_mode": str(config.get("stopping_mode", selection.MODE_CONVERGENCE)),
            "stopping_patience": int(stopper.patience),
            "eligible_epochs": eligible,
            "eligible_checkpoints_saved": len(eligible),
            "stopped_early": stopped,
            "stopped_at_ceiling": not stopped,
            "stop_reason": ("primary_plateau" if stopped else "max_epochs_ceiling"),
            "stopped_at_epoch": stopper.stopped_epoch,
            "epochs_completed": (stopper.stopped_epoch + 1) if stopped else total_epochs,
            "progress_metric": f"val_{selector.progress_metric()}",
            **guard_audit,
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
        logger.summary({"best_value": selector.value, "best_epoch": float(selector.epoch)})

    reporting.write_summary(run_dir, summary)
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
