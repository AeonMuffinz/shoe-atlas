"""The label glossary: one entry per label carrying a CLIP prompt, display names, and scoreability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GLOSSARY_PATH: Path = Path("locales/glossary.json")

TIER_DIRECT: str = "direct"
TIER_DESCRIPTIVE: str = "descriptive"
TIER_DECODED: str = "decoded"

REVIEW_PENDING: str = "pending"
REVIEW_SELF: str = "self"
REVIEW_CONFIRMED: str = "confirmed"


@dataclass
class Entry:
    label: str
    prompt: str
    display_en: str
    display_tr: str = ""
    tier: str = TIER_DIRECT
    zero_shot_scoreable: bool = True
    reason: str = ""
    review: str = REVIEW_SELF
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "display_en": self.display_en,
            "display_tr": self.display_tr,
            "tier": self.tier,
            "zero_shot_scoreable": self.zero_shot_scoreable,
            "reason": self.reason,
            "review": self.review,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, label: str, payload: dict) -> Entry:
        return cls(
            label=label,
            prompt=str(payload["prompt"]),
            display_en=str(payload.get("display_en", "")),
            display_tr=str(payload.get("display_tr", "")),
            tier=str(payload.get("tier", TIER_DIRECT)),
            zero_shot_scoreable=bool(payload.get("zero_shot_scoreable", True)),
            reason=str(payload.get("reason", "")),
            review=str(payload.get("review", REVIEW_SELF)),
            notes=str(payload.get("notes", "")),
        )


@dataclass
class Glossary:
    entries: dict[str, Entry] = field(default_factory=dict)

    def prompt(self, label: str) -> str:
        return self.entries[label].prompt

    def display(self, label: str, language: str = "en") -> str:
        entry = self.entries[label]
        return entry.display_tr if language == "tr" and entry.display_tr else entry.display_en

    def scoreable(self) -> list[str]:
        return sorted(name for name, entry in self.entries.items() if entry.zero_shot_scoreable)

    def excluded(self) -> list[str]:
        return sorted(name for name, entry in self.entries.items() if not entry.zero_shot_scoreable)

    def pending_review(self) -> list[str]:
        return sorted(name for name, entry in self.entries.items() if entry.review == REVIEW_PENDING)

    def save(self, path: Path = GLOSSARY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: entry.to_dict() for name, entry in sorted(self.entries.items())}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = GLOSSARY_PATH) -> Glossary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(entries={name: Entry.from_dict(name, body) for name, body in payload.items()})


def assert_covers(glossary: Glossary, columns: tuple[str, ...] | list[str]) -> None:
    missing = [c for c in columns if c not in glossary.entries]
    if missing:
        raise KeyError(f"{len(missing)} label(s) have no glossary entry: {missing[:8]}")


def assert_reviewed(glossary: Glossary, columns: tuple[str, ...] | list[str]) -> None:
    pending = [c for c in columns if glossary.entries[c].review == REVIEW_PENDING]
    if pending:
        raise ValueError(
            f"{len(pending)} label(s) still await review and would produce a confidently wrong "
            f"baseline: {pending}"
        )
