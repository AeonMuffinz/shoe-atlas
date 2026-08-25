"""Loads a filled-in review worksheet back into the glossary.

The other half of scripts.glossary_worksheet. Turkish comes only from the sheet a person
filled in; nothing here composes a term. Every row is validated against the glossary before
anything is written, so a sheet with a stray or missing key is refused rather than half applied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from src.catalog import FAMILIES
from src.glossary import (
    GLOSSARY_PATH,
    TR_STATUSES,
    Glossary,
    assert_covers_families,
)

DEFAULT_SHEET: Path = Path("artifacts/glossary_review_tr.csv")
REQUIRED_COLUMNS: tuple[str, ...] = ("kind", "key", "display_tr")
KIND_FAMILY: str = "family"
KIND_LABEL: str = "label"


class ImportError_(ValueError):
    pass


@dataclass(frozen=True)
class Summary:
    labels: int
    families: int
    by_status: dict[str, int]
    unchanged: int


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ImportError_(f"{path} has no rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ImportError_(f"{path} is missing required column(s) {missing}")
    return rows


def assert_sheet_matches(rows: list[dict[str, str]], glossary: Glossary) -> None:
    expected = {(KIND_FAMILY, f) for f in FAMILIES} | {(KIND_LABEL, k) for k in glossary.entries}
    got = {(r["kind"], r["key"]) for r in rows}
    if len(got) != len(rows):
        raise ImportError_("the sheet has duplicate kind/key pairs")
    if got != expected:
        extra = sorted(got - expected)
        absent = sorted(expected - got)
        raise ImportError_(
            f"the sheet does not match the glossary. {len(absent)} term(s) missing "
            f"{absent[:5]}, {len(extra)} unknown {extra[:5]}"
        )
    blank = [r["key"] for r in rows if not r["display_tr"].strip()]
    if blank:
        raise ImportError_(
            f"{len(blank)} term(s) have no Turkish name and would silently keep falling back to "
            f"English: {blank[:8]}"
        )
    unknown = sorted({r.get("status", "").strip() for r in rows} - set(TR_STATUSES) - {""})
    if unknown:
        raise ImportError_(f"unknown status value(s) {unknown}, expected one of {list(TR_STATUSES)}")


def apply_sheet(rows: list[dict[str, str]], glossary: Glossary) -> Summary:
    labels = families = unchanged = 0
    by_status: dict[str, int] = {}
    for row in rows:
        target = glossary.families if row["kind"] == KIND_FAMILY else glossary.entries
        entry = target[row["key"]]
        turkish = row["display_tr"].strip()
        if entry.display_tr == turkish:
            unchanged += 1
        entry.display_tr = turkish
        entry.tr_source = row.get("source_tr", "").strip()
        entry.tr_status = row.get("status", "").strip()
        by_status[entry.tr_status] = by_status.get(entry.tr_status, 0) + 1
        if row["kind"] == KIND_FAMILY:
            families += 1
        else:
            labels += 1
    return Summary(labels=labels, families=families, by_status=by_status, unchanged=unchanged)


def run(sheet_path: Path, glossary_path: Path) -> Summary:
    glossary = Glossary.load(glossary_path)
    assert_covers_families(glossary, FAMILIES)
    rows = read_sheet(sheet_path)
    assert_sheet_matches(rows, glossary)
    summary = apply_sheet(rows, glossary)
    glossary.save(glossary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Turkish display names from a review worksheet")
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--glossary", type=Path, default=GLOSSARY_PATH)
    args = parser.parse_args()

    summary = run(args.sheet, args.glossary)
    print(f"read    {args.sheet}")
    print(f"written {args.glossary}")
    print(f"  labels applied   {summary.labels}")
    print(f"  families applied {summary.families}")
    print(f"  already matching {summary.unchanged}")
    for status, count in sorted(summary.by_status.items()):
        print(f"  status {status or '(blank)':10s} {count}")

    reloaded = Glossary.load(args.glossary)
    print(f"  awaiting Turkish {len(reloaded.awaiting_turkish())}")


if __name__ == "__main__":
    main()
