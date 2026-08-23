"""Computes the layer_decay that matches the reference backbone floor, and measures VRAM."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from src import losses, model_builder, utils
from src.catalog import LabelSchema
from src.data_setup import load_artifacts

REFERENCE = "convnext_tiny.fb_in22k_ft_in1k"
CANDIDATES = [
    "convnext_small.fb_in22k_ft_in1k",
    "convnext_base.fb_in22k_ft_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
]
BASE_LR = 3.0e-4
REFERENCE_DECAY = 0.75
BATCH = 64


def profile(backbone: str, decay: float, num_labels: int) -> tuple[int, float]:
    cfg = model_builder.ModelConfig(
        backbone=backbone, num_labels=num_labels, pretrained=False,
        opt="adamw", lr=BASE_LR, head_lr=1e-3, weight_decay=0.05, layer_decay=decay,
    )
    model = model_builder.build_model(cfg)
    opt = model_builder.build_optimizer(model, cfg, model_builder.FULL_PHASE)
    lrs = sorted({round(float(g["lr"]), 15) for g in opt.param_groups})
    del model, opt
    return len(lrs), lrs[0]


def stress(
    backbone: str, decay: float, schema: LabelSchema, device: torch.device
) -> tuple[float, bool]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = model_builder.ModelConfig(
        backbone=backbone, num_labels=len(schema.columns), pretrained=False, drop_path=0.1,
        opt="adamw", lr=BASE_LR, head_lr=1e-3, weight_decay=0.05, layer_decay=decay,
        grad_checkpointing=True,
    )
    model = model_builder.build_model(cfg).to(device)
    opt = model_builder.build_optimizer(model, cfg, model_builder.FULL_PHASE)
    loss_fn = losses.FamilyLoss(schema, torch.ones(len(schema.columns))).to(device)
    target = torch.zeros(BATCH, len(schema.columns), device=device)
    for family in schema.softmax_families():
        target[:, family.start] = 1.0
    mask = torch.ones(BATCH, 8, dtype=torch.bool, device=device)
    ok = True
    try:
        for _ in range(10):
            images = torch.randn(BATCH, 3, 224, 224, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = loss_fn(model(images), target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        ok = False
    reserved = torch.cuda.max_memory_reserved() / 2**20
    del model, opt, loss_fn, target, mask
    torch.cuda.empty_cache()
    return reserved, ok


def main() -> None:
    utils.set_seed(42)
    schema = load_artifacts(Path("data/processed")).schema
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    free_mib = free / 2**20
    print(f"free {free_mib:,.0f} MiB of {total/2**20:,.0f}\n")

    ref_groups, ref_floor = profile(REFERENCE, REFERENCE_DECAY, len(schema.columns))
    ref_exp = ref_groups - 1
    print(f"reference {REFERENCE}")
    print(f"  distinct lrs {ref_groups}, exponent {ref_exp}, floor {ref_floor:.3e} "
          f"at layer_decay {REFERENCE_DECAY}\n")

    for backbone in CANDIDATES:
        groups, nominal_floor = profile(backbone, REFERENCE_DECAY, len(schema.columns))
        exponent = groups - 1
        matched = math.exp(math.log(REFERENCE_DECAY) * ref_exp / exponent)
        matched = round(matched, 4)
        check_groups, check_floor = profile(backbone, matched, len(schema.columns))
        print(f"{backbone}")
        print(f"  distinct lrs {groups}, exponent {exponent}")
        print(f"  at nominal layer_decay {REFERENCE_DECAY}: floor {nominal_floor:.3e}  "
              f"({nominal_floor/ref_floor:.4f}x the reference)")
        print(f"  matched layer_decay      {matched}: floor {check_floor:.3e}  "
              f"({check_floor/ref_floor:.4f}x the reference)")
        reserved, ok = stress(backbone, matched, schema, device)
        verdict = "FITS" if ok and reserved < free_mib else "DOES NOT FIT"
        print(f"  VRAM at batch {BATCH} with checkpointing: reserved {reserved:,.0f} MiB, "
              f"headroom {free_mib - reserved:,.0f} MiB -> {verdict}")
        print()


if __name__ == "__main__":
    main()
