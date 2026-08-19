"""Replays the offline guard over each recorded curve to see whether the eligible window had closed."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("artifacts/runs")
EPS = 0.01
CURVES = [
    ("convnext_tiny_s42 (legacy wiring)", RUNS / "convnext_tiny_s42" / "metrics.jsonl"),
    ("swin_tiny_s42 (legacy wiring)", RUNS / "swin_tiny_s42" / "metrics.jsonl"),
    ("convnext_tiny_s42 30-epoch (.bak)", RUNS / "convnext_tiny_s42" / "metrics_20260818_174549.jsonl.bak"),
    ("convnext_tiny_lr1_s42 (new rule)", RUNS / "convnext_tiny_lr1_s42" / "metrics.jsonl"),
    ("convnext_tiny_r160_s42 (new rule)", RUNS / "convnext_tiny_r160_s42" / "metrics.jsonl"),
    ("convnext_tiny_ld06_s42 (new rule)", RUNS / "convnext_tiny_ld06_s42" / "metrics.jsonl"),
]


def curve(path: Path) -> list[tuple[int, float, float]]:
    seen: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("phase") == "epoch" and "val_map_softmax" in record:
            seen[int(record["epoch"])] = record
    return [
        (e, float(seen[e]["val_map_softmax"]), float(seen[e]["val_map_bce"])) for e in sorted(seen)
    ]


def report(name: str, rows: list[tuple[int, float, float]]) -> dict:
    peak = max(s for _e, s, _b in rows)
    floor = (1.0 - EPS) * peak
    eligible = [(e, s, b) for e, s, b in rows if s >= floor]
    last_epoch = rows[-1][0]
    last_eligible = eligible[-1][0] if eligible else None
    window_closed = last_eligible is not None and last_eligible < last_epoch

    print(f"\n{'=' * 84}\n{name}\n{'=' * 84}")
    print(f"  epochs recorded 0..{last_epoch}   guard peak {peak:.4f}   floor {floor:.4f}")
    print(f"  {'epoch':>6}{'softmax':>10}{'map_bce':>10}   eligible")
    for e, s, b in rows:
        ok = s >= floor
        mark = "  YES" if ok else ""
        if ok or e >= last_epoch - 3:
            print(f"  {e:>6}{s:>10.4f}{b:>10.4f}{mark}")
    print(f"\n  eligible epochs: {[e for e, _s, _b in eligible]}")
    print(f"  map_bce at each: {[round(b, 4) for _e, _s, b in eligible]}")
    print(f"  best eligible  : epoch {max(eligible, key=lambda x: x[2])[0]} "
          f"at map_bce {max(b for _e, _s, b in eligible):.4f}")
    print(f"  last eligible epoch {last_eligible}, last recorded epoch {last_epoch}")

    gaps = []
    run = 0
    for _e, s, _b in rows:
        if s >= floor:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    print(f"  recovery gaps (consecutive ineligible epochs that were followed by an eligible one): "
          f"{gaps if gaps else 'none'}")
    print(f"  trailing ineligible epochs at the end of the run: {run}")

    if window_closed:
        print(f"  VERDICT: window CLOSED {last_epoch - last_eligible} epochs before the run ended. "
              f"The recorded selection is final.")
    else:
        print("  VERDICT: the final recorded epoch is STILL ELIGIBLE. "
              "The selection is a lower bound, not a value.")
    return {"gaps": gaps, "trailing": run, "closed": window_closed}


def main() -> None:
    all_gaps: list[int] = []
    for name, path in CURVES:
        if not path.exists():
            print(f"\n{name}: missing")
            continue
        result = report(name, curve(path))
        all_gaps.extend(result["gaps"])
    print(f"\n{'=' * 84}\nrecovery gaps observed across every curve: {sorted(all_gaps)}")
    print(f"largest gap that still recovered: {max(all_gaps) if all_gaps else 0}")


if __name__ == "__main__":
    main()
