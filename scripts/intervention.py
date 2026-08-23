"""Scores an intervention run against the reference on the pre-registered criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path("artifacts/runs")
REFERENCE = "convnext_tiny_s42"
LONG_REFERENCE = RUNS / "convnext_tiny_s42" / "metrics_20260818_174549.jsonl.bak"
FLOOR = 0.001


def epochs(path: Path) -> list[dict]:
    seen: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("phase") == "epoch" and "val_map" in record:
            seen[int(record["epoch"])] = record
    return [seen[k] for k in sorted(seen)]


def summarise(rows: list[dict]) -> dict:
    soft = [(int(r["epoch"]), float(r["val_map_softmax"])) for r in rows]
    bce = [(int(r["epoch"]), float(r["val_map_bce"])) for r in rows]
    peak_epoch, peak = max(soft, key=lambda x: x[1])
    final_epoch, final = soft[-1]
    after = [v for e, v in soft if e > peak_epoch]
    return {
        "epochs": len(rows),
        "soft_peak": peak,
        "soft_peak_epoch": peak_epoch,
        "soft_final": final,
        "soft_final_epoch": final_epoch,
        "decline_to_end": peak - final,
        "worst_decline": peak - min(after) if after else 0.0,
        "bce_peak": max(v for _e, v in bce),
        "bce_final": bce[-1][1],
        "bce_still_rising": bce[-1][1] >= max(v for _e, v in bce) - 1e-12,
    }


def curve(rows: list[dict]) -> tuple[list[float], list[float]]:
    return ([float(r["val_map_bce"]) for r in rows], [float(r["val_map_softmax"]) for r in rows])


def matched_progress(ref: list[dict], new: list[dict]) -> list[tuple[float, float, float]]:
    rb, rs = curve(ref)
    nb, ns = curve(new)
    low, high = max(min(rb), min(nb)), min(max(rb), max(nb))
    if high <= low:
        return []
    levels = np.linspace(low, high, 12)
    return [(float(x), float(np.interp(x, rb, rs)), float(np.interp(x, nb, ns))) for x in levels]


def table(rows: list[dict], title: str) -> None:
    print(f"\n{title}   {len(rows)} epochs")
    print(f"  {'ep':>3}{'train':>9}{'val_loss':>10}{'map':>9}{'soft':>9}{'bce':>9}")
    for r in rows:
        print(f"  {int(r['epoch']):>3}{float(r['train_loss']):>9.4f}{float(r['val_loss']):>10.4f}"
              f"{float(r['val_map']):>9.4f}{float(r['val_map_softmax']):>9.4f}"
              f"{float(r['val_map_bce']):>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare an intervention run against the reference")
    parser.add_argument("run", type=str)
    parser.add_argument("--baseline", type=str, default=REFERENCE)
    args = parser.parse_args()

    new_rows = epochs(RUNS / args.run / "metrics.jsonl")
    ref_rows = epochs(RUNS / args.baseline / "metrics.jsonl")
    long_path = LONG_REFERENCE if args.baseline == REFERENCE else RUNS / args.baseline / "metrics.jsonl"
    long_rows = epochs(long_path)

    table(ref_rows, f"REFERENCE {args.baseline}")
    table(new_rows, f"INTERVENTION {args.run}")

    ref, new = summarise(ref_rows), summarise(new_rows)
    print(f"\n{'':<26}{'reference':>12}{'intervention':>14}{'delta':>10}")
    for key in ("epochs", "soft_peak", "soft_peak_epoch", "soft_final", "decline_to_end",
                "worst_decline", "bce_peak", "bce_final"):
        a, b = ref[key], new[key]
        delta = b - a
        fmt = "d" if isinstance(a, int) else ".4f"
        print(f"{key:<26}{a:>12{fmt}}{b:>14{fmt}}{delta:>+10{fmt}}")

    print("\nPRE-REGISTERED CRITERIA")
    c1 = new["soft_peak"] - ref["soft_peak"]
    print(f"  1. guard_peak rises by > {FLOOR}      : {c1:+.4f}  -> "
          f"{'EFFECT' if c1 > FLOOR else 'no'}")
    c3 = ref["decline_to_end"] - new["decline_to_end"]
    print(f"  3. decline shrinks by > {FLOOR}       : {c3:+.4f}  -> "
          f"{'EFFECT' if c3 > FLOOR else 'no'}")
    print("  2. WITHDRAWN (epoch-indexed peak delay is confounded by training speed)")

    print("\nMATCHED-PROGRESS TRADE-OFF (replaces criterion 2)")
    print(f"  {'bce level':>10}{'ref soft':>10}{'new soft':>10}{'delta':>9}")
    deltas = []
    for level, r, n in matched_progress(long_rows, new_rows):
        deltas.append(n - r)
        print(f"  {level:>10.4f}{r:>10.4f}{n:>10.4f}{n - r:>+9.4f}")
    if deltas:
        arr = np.array(deltas)
        print(f"  mean {arr.mean():+.4f}   max {arr.max():+.4f}   min {arr.min():+.4f}")
        verdict = "EFFECT" if arr.mean() > FLOOR else ("REVERSED" if arr.mean() < -FLOOR else "NULL")
        print(f"  -> {verdict} against the {FLOOR} floor "
              f"(compared against the 30-epoch reference curve)")


if __name__ == "__main__":
    main()
