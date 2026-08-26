"""Builds the cross-run comparison table from the persisted run artifacts. Reads JSON, needs no torch."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src import reporting
from src.reporting import RUNS_ROOT, WINNER_PATH

COMPARISON_MD: Path = Path("artifacts/comparison.md")
COMPARISON_CSV: Path = Path("artifacts/comparison.csv")
NOT_MEASURED: str = "not measured"
CAPACITY_PROBE: str = "capacity_probe"
SEED_SUFFIX = re.compile(r"_s\d+$")
MIN_SEEDS_FOR_SPREAD: int = 3

COLUMNS: tuple[tuple[str, str], ...] = (
    ("map", "mAP"),
    ("map_softmax", "mAP softmax"),
    ("map_bce", "mAP bce"),
    ("macro_f1_bce_out_of_fold", "macroF1 oof"),
    ("calibration_error_out_of_fold", "ECE oof"),
)
FAMILIES: tuple[str, ...] = ("Category", "SubCategory", "HeelHeight")


@dataclass
class Run:
    name: str
    summary: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return SEED_SUFFIX.sub("", self.name)

    @property
    def role(self) -> str:
        return str(self.summary.get("role", self.evaluation.get("role", NOT_MEASURED)))

    @property
    def val(self) -> dict:
        block = self.evaluation.get("val")
        return block if isinstance(block, dict) else {}

    def metric(self, key: str) -> float | None:
        value = self.val.get(key)
        return float(value) if isinstance(value, int | float) else None

    def family_top1(self, family: str) -> float | None:
        block = self.val.get("family_top1")
        if not isinstance(block, dict):
            return None
        value = block.get(family)
        return float(value) if isinstance(value, int | float) else None

    @property
    def confounds(self) -> list[str]:
        block = self.summary.get("capacity_probe")
        if not isinstance(block, dict):
            return []
        return [str(c) for c in block.get("confounds", [])]

    @property
    def capacity_reference(self) -> str:
        block = self.summary.get("capacity_probe")
        return str(block.get("reference_run", NOT_MEASURED)) if isinstance(block, dict) else NOT_MEASURED

    def test_cell(self, winner: str | None) -> str:
        scores = self.evaluation.get("test")
        if isinstance(scores, dict):
            return cell(scores.get("map"))
        if winner is not None and self.name == winner:
            return "winner, test not yet scored"
        return reporting.TEST_WITHHELD


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def discover(runs_root: Path = RUNS_ROOT) -> tuple[list[Run], list[str]]:
    live: list[Run] = []
    archived: list[str] = []
    for directory in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        summary = load_json(directory / reporting.SUMMARY_NAME)
        if reporting.is_archived(summary):
            archived.append(directory.name)
            continue
        evaluation = load_json(directory / reporting.EVALUATION_NAME)
        if not summary and not evaluation:
            continue
        live.append(Run(name=directory.name, summary=summary, evaluation=evaluation))
    return live, archived


def read_winner(path: Path = WINNER_PATH) -> str | None:
    payload = load_json(path)
    name = payload.get("winner")
    return str(name) if name else None


def cell(value: float | None, digits: int = 4) -> str:
    return NOT_MEASURED if value is None else f"{value:.{digits}f}"


def epochs_cell(run: Run) -> str:
    completed = run.summary.get("epochs_completed")
    if completed is None:
        return NOT_MEASURED
    if run.summary.get("stopped_at_ceiling") is True:
        return f"{int(completed)} (ceiling)"
    if run.summary.get("stopped_early") is True:
        return f"{int(completed)} (plateau)"
    return str(int(completed))


def guard_cell(run: Run) -> str:
    rejected = run.summary.get("epochs_rejected_by_guard")
    if rejected is None:
        return NOT_MEASURED
    peak = run.summary.get("guard_peak")
    at_best = run.summary.get("guard_at_best")
    if isinstance(peak, int | float) and isinstance(at_best, int | float):
        return f"{int(rejected)} rejected, peak {peak:.4f}, at best {at_best:.4f}"
    return f"{int(rejected)} rejected"


def audit_cell(run: Run) -> str:
    matches = run.summary.get("online_matches_offline")
    if matches is None:
        return NOT_MEASURED
    offline = run.summary.get("offline_selected_epoch")
    return f"agrees (ep {offline})" if matches else f"DIVERGED, offline ep {offline}"


def macro_f1_note(run: Run) -> str:
    basis = run.val.get("macro_f1_basis")
    return f" ({basis})" if isinstance(basis, str) else ""


def rows_for_csv(runs: list[Run], winner: str | None) -> list[dict[str, str]]:
    out = []
    for run in runs:
        row = {
            "run": run.name,
            "stem": run.stem,
            "role": run.role,
            "selected_epoch": str(run.evaluation.get("epoch", NOT_MEASURED)),
            "epochs_completed": epochs_cell(run),
            "stop_reason": str(run.summary.get("stop_reason", NOT_MEASURED)),
            "guard": guard_cell(run),
            "selection_audit": audit_cell(run),
            "test": run.test_cell(winner),
        }
        for key, _label in COLUMNS:
            row[key] = cell(run.metric(key))
        for family in FAMILIES:
            row[f"top1_{family}"] = cell(run.family_top1(family))
        out.append(row)
    return out


def seed_groups(runs: list[Run]) -> dict[str, list[Run]]:
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.stem, []).append(run)
    return {stem: members for stem, members in grouped.items() if len(members) >= MIN_SEEDS_FOR_SPREAD}


def mean_and_spread(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, variance**0.5


def split_by_role(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    grid = [r for r in runs if r.role != CAPACITY_PROBE]
    capacity = [r for r in runs if r.role == CAPACITY_PROBE]
    return grid, capacity


def metric_table(runs: list[Run], winner: str | None) -> list[str]:
    header = ["run", "role", "sel ep", "epochs"] + [label for _key, label in COLUMNS]
    header += [f"top1 {f}" for f in FAMILIES] + ["test"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for run in runs:
        cells = [
            f"`{run.name}`",
            run.role,
            str(run.evaluation.get("epoch", NOT_MEASURED)),
            epochs_cell(run),
        ]
        for key, _label in COLUMNS:
            suffix = macro_f1_note(run) if key == "macro_f1_bce_out_of_fold" else ""
            cells.append(cell(run.metric(key)) + suffix)
        cells += [cell(run.family_top1(f)) for f in FAMILIES]
        cells.append(run.test_cell(winner))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(runs: list[Run], archived: list[str], winner: str | None) -> str:
    lines: list[str] = ["# Run comparison", ""]
    lines.append(
        "Validation figures for every live run. Selection is made on validation, so this is the table "
        "the winner is chosen from."
    )
    lines.append("")
    if winner is None:
        lines.append(
            f"**No winner recorded.** `{WINNER_PATH}` does not exist, so the test set has never been "
            "unlocked and every test cell below says so."
        )
    else:
        lines.append(f"**Winner: `{winner}`.** Only that row may carry a test score.")
    lines.append("")

    grid, capacity = split_by_role(runs)
    lines += metric_table(grid, winner)

    if capacity:
        lines += ["", "## Capacity probes, reported separately from the grid", ""]
        lines.append(
            "The comparison grid holds every candidate at similar capacity so that architecture is not "
            "confounded with parameter count. A larger backbone breaks that premise by design, so these "
            "runs are reported here rather than as another architecture row."
        )
        lines.append("")
        lines += metric_table(capacity, winner)
        lines.append("")
        for run in capacity:
            lines.append(f"**`{run.name}`** against `{run.capacity_reference}`. Confounds:")
            for confound in run.confounds:
                lines.append(f"- {confound}")
            if not run.confounds:
                lines.append(f"- {NOT_MEASURED}")
            lines.append("")

    lines += ["", "## Selection and guard", ""]
    lines.append("| run | stop reason | guard | online vs offline |")
    lines.append("|---|---|---|---|")
    for run in runs:
        lines.append(
            f"| `{run.name}` | {run.summary.get('stop_reason', NOT_MEASURED)} | "
            f"{guard_cell(run)} | {audit_cell(run)} |"
        )

    groups = seed_groups(runs)
    lines += ["", "## Seed spread", ""]
    if not groups:
        lines.append(
            f"No stem has {MIN_SEEDS_FOR_SPREAD} seeds yet, so no mean or spread is reported. "
            "The tie-break rule needs seed spread before it can be applied."
        )
    for stem, members in sorted(groups.items()):
        lines.append(f"**`{stem}`**, {len(members)} seeds")
        for key, label in COLUMNS:
            values = [m.metric(key) for m in members]
            present = [v for v in values if v is not None]
            if len(present) < MIN_SEEDS_FOR_SPREAD:
                lines.append(f"- {label}: {NOT_MEASURED}")
                continue
            mean, spread = mean_and_spread(present)
            lines.append(f"- {label}: {mean:.4f} ± {spread:.4f}")
        lines.append("")

    if archived:
        lines += ["", "## Archived, excluded from the table above", ""]
        for name in archived:
            lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def build(runs_root: Path = RUNS_ROOT, md: Path = COMPARISON_MD, csv_path: Path = COMPARISON_CSV) -> str:
    runs, archived = discover(runs_root)
    winner = read_winner()
    text = render(runs, archived, winner)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text, encoding="utf-8")

    rows = rows_for_csv(runs, winner)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the cross-run comparison table")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--markdown", type=Path, default=COMPARISON_MD)
    parser.add_argument("--csv", type=Path, default=COMPARISON_CSV)
    args = parser.parse_args()
    print(build(args.runs_root, args.markdown, args.csv))


if __name__ == "__main__":
    main()
