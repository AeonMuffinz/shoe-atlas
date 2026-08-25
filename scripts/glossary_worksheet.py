"""Writes the review worksheet the operator fills in with Turkish display names.

Turkish terms are never generated here. Every display_tr column is left as it stands
in the glossary, which is empty until a person fills it in.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.catalog import FAMILIES
from src.glossary import (
    GLOSSARY_PATH,
    TIER_DECODED,
    TIER_DESCRIPTIVE,
    Glossary,
    assert_covers_families,
)

OUT_PATH: Path = Path("artifacts/glossary_review.csv")

NAMED_TRAPS: dict[str, str] = {
    "Closure": "the design notes say this is NOT a literal rendering of the English",
    "ToeStyle": "the design notes say this is NOT a literal rendering of the English",
}

RECORDED_CEILINGS: dict[str, str] = {
    "ToeStyle.Moc.Toe": "differs from Algonquin only in whether the same U-seam is puckered or flat",
    "ToeStyle.Algonquin": "near-identical to Moc Toe; a people's name in English, not a foot shape",
    "ToeStyle.Medallion": "a perforated pattern on the toe cap, not a pendant",
    "Material.Hair.Calf": "calfskin with the hair left on; not literally hair",
    "Material.Terry": "91% concentrated in slipper flats, so a hit may be recognising the slipper",
    "SubCategory.Prewalker": "pre-walking infant shoe, soft sole",
    "SubCategory.Firstwalker": "first walking shoe, stiffer sole than a prewalker",
}

CARE_RANK: dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "low": 2}
FIELDS: tuple[str, ...] = (
    "kind", "family", "key", "display_en", "display_tr", "tier", "review",
    "zero_shot_scoreable", "care", "context", "notes",
)


def care_for(key: str, tier: str) -> str:
    if key in RECORDED_CEILINGS:
        return f"HIGH - {RECORDED_CEILINGS[key]}"
    if key in NAMED_TRAPS:
        return f"HIGH - {NAMED_TRAPS[key]}"
    if tier == TIER_DECODED:
        return "HIGH - hand-decoded range, not a literal phrase"
    if tier == TIER_DESCRIPTIVE:
        return "MEDIUM - the English name misleads, translate the described thing"
    return "low"


def family_rows(glossary: Glossary) -> list[dict[str, object]]:
    rows = []
    for family in FAMILIES:
        entry = glossary.families[family]
        rows.append({
            "kind": "family",
            "family": family,
            "key": family,
            "display_en": entry.display_en,
            "display_tr": entry.display_tr,
            "tier": "",
            "review": "",
            "zero_shot_scoreable": "",
            "care": care_for(family, ""),
            "context": "shown as the group heading in the interface",
            "notes": entry.notes,
        })
    return rows


def label_rows(glossary: Glossary) -> list[dict[str, object]]:
    rows = []
    for key, entry in sorted(glossary.entries.items()):
        rows.append({
            "kind": "label",
            "family": key.split(".", 1)[0],
            "key": key,
            "display_en": entry.display_en,
            "display_tr": entry.display_tr,
            "tier": entry.tier,
            "review": entry.review,
            "zero_shot_scoreable": entry.zero_shot_scoreable,
            "care": care_for(key, entry.tier),
            "context": entry.prompt,
            "notes": entry.notes or entry.reason,
        })
    return rows


def sort_key(row: dict[str, object]) -> tuple[int, int, int, str]:
    care = str(row["care"]).split(" -")[0]
    return (FAMILIES.index(str(row["family"])), int(row["kind"] == "label"),
            CARE_RANK[care], str(row["key"]))


def build(glossary_path: Path = GLOSSARY_PATH) -> list[dict[str, object]]:
    glossary = Glossary.load(glossary_path)
    assert_covers_families(glossary, FAMILIES)
    rows = family_rows(glossary) + label_rows(glossary)
    rows.sort(key=sort_key)
    return rows


def write(rows: list[dict[str, object]], out_path: Path = OUT_PATH) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    rows = build()
    out_path = write(rows)
    families = sum(1 for r in rows if r["kind"] == "family")
    filled = sum(1 for r in rows if str(r["display_tr"]).strip())
    high = sum(1 for r in rows if str(r["care"]).startswith("HIGH"))
    medium = sum(1 for r in rows if str(r["care"]).startswith("MEDIUM"))
    print(f"written {out_path}")
    print(f"  rows              {len(rows)} ({families} families + {len(rows) - families} labels)")
    print(f"  display_tr filled {filled} of {len(rows)}")
    print(f"  care HIGH         {high}")
    print(f"  care MEDIUM       {medium}")


if __name__ == "__main__":
    main()
