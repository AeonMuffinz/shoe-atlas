"""User-facing strings, one JSON per language. Turkish is the default and nothing is hardcoded.

Deliberately not gr.I18n: that keys off browser language, and the requirement is a Turkish default
with an explicit switch. Keeping the dictionaries here also makes the no-hardcoded-strings rule
checkable by a test rather than by reading the interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LOCALES_DIR: Path = Path("locales")
DEFAULT_LANGUAGE: str = "tr"
LANGUAGES: tuple[str, ...] = ("tr", "en")
LANGUAGE_NAMES: dict[str, str] = {"tr": "Türkçe", "en": "English"}


class LocaleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Locale:
    language: str
    strings: dict[str, str]

    def text(self, key: str, **fields: object) -> str:
        if key not in self.strings:
            raise LocaleError(
                f"{self.language} has no string for {key!r}. Every user-facing string comes from a "
                "locale file, so a missing key is a bug rather than a reason to hardcode one."
            )
        try:
            return self.strings[key].format(**fields)
        except KeyError as exc:
            raise LocaleError(
                f"{key!r} in {self.language} wants a field this call did not pass: {exc}"
            ) from exc


def locale_path(language: str, root: Path = LOCALES_DIR) -> Path:
    return root / f"{language}.json"


def load_locale(language: str, root: Path = LOCALES_DIR) -> Locale:
    if language not in LANGUAGES:
        raise LocaleError(f"unknown language {language!r}, expected one of {list(LANGUAGES)}")
    path = locale_path(language, root)
    if not path.exists():
        raise LocaleError(f"{path} is missing")
    return Locale(language=language, strings=json.loads(path.read_text(encoding="utf-8")))


def load_all(root: Path = LOCALES_DIR) -> dict[str, Locale]:
    return {language: load_locale(language, root) for language in LANGUAGES}


def assert_languages_agree(locales: dict[str, Locale]) -> None:
    keysets = {language: set(locale.strings) for language, locale in locales.items()}
    reference = keysets[DEFAULT_LANGUAGE]
    for language, keys in keysets.items():
        missing = sorted(reference - keys)
        extra = sorted(keys - reference)
        if missing or extra:
            raise LocaleError(
                f"{language} and {DEFAULT_LANGUAGE} do not carry the same keys. Missing {missing[:5]}, "
                f"extra {extra[:5]}. A key present in one language and not the other shows as English "
                "in a Turkish interface, which is the half-finished job the design forbids."
            )


def assert_no_blank_strings(locales: dict[str, Locale]) -> None:
    for language, locale in locales.items():
        blank = sorted(key for key, value in locale.strings.items() if not str(value).strip())
        if blank:
            raise LocaleError(f"{language} has {len(blank)} blank string(s): {blank[:5]}")
