"""Writes the review worksheet the operator fills in with Turkish display names.

Turkish terms are never generated here. Every display_tr column is left as it stands
in the glossary, which is empty until a person fills it in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.catalog import FAMILIES
from src.glossary import GLOSSARY_PATH, TIER_DECODED, TIER_DESCRIPTIVE

OUT_PATH: Path = Path("artifacts/glossary_review.csv")

FAMILY_EN: dict[str, str] = {
    "Category": "Category",
    "SubCategory": "Sub-category",
    "HeelHeight": "Heel height",
    "Insole": "Insole",
    "Closure": "Closure",
    "Gender": "Gender",
    "Material": "Material",
    "ToeStyle": "Toe style",
}

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

FAMILY_MISSING: str = "MISSING - no glossary entry exists for the family heading"
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


def family_rows() -> list[dict[str, object]]:
    return [
        {
            "kind": "family",
            "family": family,
            "key": family,
            "display_en": FAMILY_EN[family],
            "display_tr": "",
            "tier": "",
            "review": FAMILY_MISSING,
            "zero_shot_scoreable": "",
            "care": care_for(family, ""),
            "context": "shown as the group heading in the interface",
            "notes": "",
        }
        for family in FAMILIES
    ]


def label_rows(glossary: dict[str, dict]) -> list[dict[str, object]]:
    rows = []
    for key, entry in sorted(glossary.items()):
        tier = str(entry.get("tier", ""))
        rows.append({
            "kind": "label",
            "family": key.split(".", 1)[0],
            "key": key,
            "display_en": entry.get("display_en", ""),
            "display_tr": entry.get("display_tr", ""),
            "tier": tier,
            "review": entry.get("review", ""),
            "zero_shot_scoreable": entry.get("zero_shot_scoreable", ""),
            "care": care_for(key, tier),
            "context": entry.get("prompt", ""),
            "notes": entry.get("notes", "") or entry.get("reason", ""),
        })
    return rows


def sort_key(row: dict[str, object]) -> tuple[int, int, int, str]:
    care = str(row["care"]).split(" -")[0]
    return (FAMILIES.index(str(row["family"])), int(row["kind"] == "label"),
            CARE_RANK[care], str(row["key"]))


def build(glossary_path: Path = GLOSSARY_PATH) -> list[dict[str, object]]:
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    rows = family_rows() + label_rows(glossary)
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
