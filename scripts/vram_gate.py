"""Full training-step VRAM measurement for the capacity candidates, optimizer state included.

Proves the figure covers AdamW moments by reporting memory before any step, after the first,
and after thirty, alongside the count of optimizer state tensors actually allocated.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src import losses, model_builder, utils
from src.catalog import LabelSchema
from src.data_setup import load_artifacts

CANDIDATES = [
    ("convnext_tiny.fb_in22k_ft_in1k", 0.75),
    ("convnext_small.fb_in22k_ft_in1k", 0.8627),
    ("convnext_base.fb_in22k_ft_in1k", 0.8627),
]
BATCH = 64
STEPS = 30
HEADROOM_GATE = 600.0
REFERENCE_FLOOR = 1.268e-06


def measure(backbone: str, decay: float, schema: LabelSchema, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = model_builder.ModelConfig(
        backbone=backbone, num_labels=len(schema.columns), pretrained=False, drop_path=0.1,
        opt="adamw", lr=3.0e-4, head_lr=1.0e-3, weight_decay=0.05, layer_decay=decay,
        grad_checkpointing=True,
    )
    model = model_builder.build_model(cfg).to(device)
    optimizer = model_builder.build_optimizer(model, cfg, model_builder.FULL_PHASE)
    lrs = sorted({round(float(g["lr"]), 15) for g in optimizer.param_groups})
    loss_fn = losses.FamilyLoss(schema, torch.ones(len(schema.columns))).to(device)

    target = torch.zeros(BATCH, len(schema.columns), device=device)
    for family in schema.softmax_families():
        target[:, family.start] = 1.0
    mask = torch.ones(BATCH, 8, dtype=torch.bool, device=device)

    torch.cuda.synchronize()
    before = torch.cuda.max_memory_reserved() / 2**20
    after_first = None
    ok = True
    try:
        for step in range(STEPS):
            images = torch.randn(BATCH, 3, 224, 224, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = loss_fn(model(images), target, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 0:
                torch.cuda.synchronize()
                after_first = torch.cuda.max_memory_reserved() / 2**20
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        ok = False
    reserved = torch.cuda.max_memory_reserved() / 2**20

    moments = sum(
        1
        for state in optimizer.state.values()
        for key in ("exp_avg", "exp_avg_sq")
        if isinstance(state.get(key), torch.Tensor)
    )
    moment_mib = sum(
        state[key].numel() * state[key].element_size()
        for state in optimizer.state.values()
        for key in ("exp_avg", "exp_avg_sq")
        if isinstance(state.get(key), torch.Tensor)
    ) / 2**20
    params = sum(p.numel() for p in model.parameters())
    del model, optimizer, loss_fn, target, mask
    torch.cuda.empty_cache()
    return {
        "params": params, "before": before, "after_first": after_first, "reserved": reserved,
        "ok": ok, "moment_tensors": moments, "moment_mib": moment_mib, "floor": lrs[0],
    }


def main() -> None:
    utils.set_seed(42)
    device = torch.device("cuda")
    schema = load_artifacts(Path("data/processed")).schema
    free, total = torch.cuda.mem_get_info()
    free_mib = free / 2**20
    print(f"free {free_mib:,.0f} MiB of {total/2**20:,.0f}")
    print(f"batch {BATCH}, {STEPS} full steps, grad_checkpointing on, bfloat16 autocast\n")

    for backbone, decay in CANDIDATES:
        r = measure(backbone, decay, schema, device)
        headroom = free_mib - r["reserved"]
        gate = "RUNS" if headroom >= HEADROOM_GATE else "BLOCKED"
        drift = abs(r["floor"] - REFERENCE_FLOOR) / REFERENCE_FLOOR * 100
        print(f"{backbone}  (layer_decay {decay})")
        print(f"  params {r['params']:,}   backbone floor {r['floor']:.3e} "
              f"({drift:.2f}% from the reference)")
        print(f"  optimizer state: {r['moment_tensors']} moment tensors, {r['moment_mib']:,.0f} MiB")
        print(f"  reserved before any step {r['before']:,.0f} MiB")
        print(f"  reserved after step 1    {r['after_first']:,.0f} MiB   <- moments now allocated")
        print(f"  reserved after {STEPS} steps  {r['reserved']:,.0f} MiB")
        print(f"  headroom {headroom:,.0f} MiB against a {HEADROOM_GATE:,.0f} MiB gate -> "
              f"{gate}{'' if r['ok'] else '  (OOM raised)'}\n")


if __name__ == "__main__":
    main()
