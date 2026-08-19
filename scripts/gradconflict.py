"""Cosine similarity between the two head groups' trunk gradients, per parameter block."""

from __future__ import annotations

from pathlib import Path

import torch

from src import data_setup, evaluate, losses
from src.catalog import LabelSchema

PROCESSED = Path("data/processed")
RUNS = Path("artifacts/runs")
TARGETS = [
    "convnext_tiny_s42",
    "swin_tiny_s42",
    "convnext_tiny_lr1_s42",
    "convnext_tiny_r160_s42",
    "convnext_tiny_ld06_s42",
]
BATCH = 32


def block_of(name: str) -> str:
    parts = name.split(".")
    if parts[0] in ("head", "norm_pre", "norm"):
        return "head"
    if parts[0] in ("stem", "patch_embed"):
        return "stem"
    if parts[0] in ("stages", "layers") and len(parts) > 1:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def grads_by_block(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    blocks: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        blocks.setdefault(block_of(name), []).append(param.grad.detach().flatten().float())
    return {k: torch.cat(v) for k, v in blocks.items()}


def one_batch(
    artifacts: data_setup.Artifacts, cfg: data_setup.DataConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = data_setup.build_eval_dataloader(cfg, artifacts, "val")
    images, target, mask = next(iter(loader))
    return images[:BATCH].to(device), target[:BATCH].to(device), mask[:BATCH].to(device)


def conflict(run: str, artifacts: data_setup.Artifacts, schema: LabelSchema,
             device: torch.device) -> dict[str, float]:
    payload = evaluate.load_checkpoint(RUNS / run, "best.pt")
    model = evaluate.build_from_checkpoint(payload, len(schema.columns), device)
    model.eval()
    config = dict(payload["config"])
    cfg = data_setup.DataConfig(
        processed_dir=PROCESSED,
        image_size=int(config.get("image_size", 224)),
        batch_size=BATCH, num_workers=0, seed=42,
        mean=tuple(config.get("mean", data_setup.IMAGENET_MEAN)),
        std=tuple(config.get("std", data_setup.IMAGENET_STD)),
    )
    images, target, mask = one_batch(artifacts, cfg, device)
    pos_weight = losses.compute_pos_weight(
        artifacts.labels, artifacts.family_observed, schema,
        artifacts.splits["train"], float(config.get("pos_weight_cap", 10.0)),
    )
    soft_only = losses.FamilyLoss(
        schema, pos_weight, losses.LossConfig(softmax_weight=1.0, bce_weight=0.0)
    ).to(device)
    bce_only = losses.FamilyLoss(
        schema, pos_weight, losses.LossConfig(softmax_weight=0.0, bce_weight=1.0)
    ).to(device)

    collected = {}
    for tag, loss_fn in (("softmax", soft_only), ("bce", bce_only)):
        model.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model(images), target, mask)
        loss.backward()
        collected[tag] = grads_by_block(model)
    model.zero_grad(set_to_none=True)

    shared = sorted(set(collected["softmax"]) & set(collected["bce"]))
    cosines = {}
    for block in shared:
        a, b = collected["softmax"][block], collected["bce"][block]
        denom = a.norm() * b.norm()
        cosines[block] = float(torch.dot(a, b) / denom) if float(denom) > 0 else float("nan")
    del model
    torch.cuda.empty_cache()
    return cosines


def main() -> None:
    artifacts = data_setup.load_artifacts(PROCESSED)
    schema = artifacts.schema
    device = torch.device("cuda")
    print(f"cosine(softmax-grad, bce-grad) per block, one validation batch of {BATCH}, float32\n")
    table: dict[str, dict[str, float]] = {}
    for run in TARGETS:
        if not (RUNS / run / "best.pt").exists():
            print(f"{run}: no checkpoint")
            continue
        table[run] = conflict(run, artifacts, schema, device)

    blocks = sorted({b for row in table.values() for b in row})
    width = max(len(b) for b in blocks) + 2
    print(f"{'block':<{width}}" + "".join(f"{r.replace('_s42',''):>24}" for r in table))
    for block in blocks:
        cells = "".join(f"{table[r].get(block, float('nan')):>24.4f}" for r in table)
        print(f"{block:<{width}}{cells}")


if __name__ == "__main__":
    main()
