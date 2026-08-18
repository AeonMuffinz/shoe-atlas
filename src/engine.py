"""The training and evaluation loops. Knows nothing about which model it is running."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.utils import RunLogger

DEFAULT_LOG_EVERY: int = 50


@dataclass
class EvalOutputs:
    metrics: dict[str, float]
    logits: Tensor
    targets: Tensor
    mask: Tensor


class RunningMean:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count: int = 0

    def update(self, components: dict[str, Tensor]) -> None:
        for key, value in components.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
        self.count += 1

    def result(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: total / self.count for key, total in self.totals.items()}


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


def move_batch(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    images, targets, mask = batch
    return (
        images.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, ...]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    logger: RunLogger,
    device: torch.device,
    epoch: int,
    accum_steps: int = 1,
    max_grad_norm: float | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    phase: str = "train",
) -> dict[str, float]:
    model.train()
    running = RunningMean()
    steps_per_epoch = len(loader) if hasattr(loader, "__len__") else None
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(logger.progress(loader, desc=f"{phase} {epoch}", total=steps_per_epoch)):
        images, targets, mask = move_batch(batch, device)
        with autocast_context(device, amp_dtype, use_amp):
            predictions = model(images)
            loss, components = loss_fn(predictions, targets, mask)
        (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        running.update(components)
        if log_every > 0 and step % log_every == 0:
            snapshot = {key: float(value) for key, value in components.items()}
            snapshot["lr"] = float(optimizer.param_groups[0]["lr"])
            logger.log(snapshot, step=epoch * (steps_per_epoch or 1) + step, phase=phase)

    end_step = (epoch + 1) * (steps_per_epoch or 1)
    summary = running.result()
    summary["lr"] = float(optimizer.param_groups[0]["lr"])
    summary["epoch"] = float(epoch)
    summary["global_step"] = float(end_step)
    logger.log(summary, step=end_step, phase=f"{phase}_epoch")
    return summary


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, ...]],
    loss_fn: nn.Module,
    logger: RunLogger,
    device: torch.device,
    epoch: int,
    phase: str = "val",
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
    global_step: int | None = None,
) -> EvalOutputs:
    model.eval()
    running = RunningMean()
    batches = len(loader) if hasattr(loader, "__len__") else None
    logit_chunks: list[Tensor] = []
    target_chunks: list[Tensor] = []
    mask_chunks: list[Tensor] = []

    for batch in logger.progress(loader, desc=f"{phase} {epoch}", total=batches):
        images, targets, mask = move_batch(batch, device)
        with autocast_context(device, amp_dtype, use_amp):
            predictions = model(images)
            _, components = loss_fn(predictions, targets, mask)
        running.update(components)
        logit_chunks.append(predictions.detach().float().cpu())
        target_chunks.append(targets.detach().cpu())
        mask_chunks.append(mask.detach().cpu())

    metrics = running.result()
    metrics["epoch"] = float(epoch)
    logger.log(metrics, step=epoch if global_step is None else global_step, phase=phase)
    return EvalOutputs(
        metrics=metrics,
        logits=torch.cat(logit_chunks) if logit_chunks else torch.empty(0),
        targets=torch.cat(target_chunks) if target_chunks else torch.empty(0),
        mask=torch.cat(mask_chunks) if mask_chunks else torch.empty(0),
    )
