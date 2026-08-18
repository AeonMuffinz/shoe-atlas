"""Assembles the joined UT Zappos50K catalog and the label schema everything downstream slices by.

Deliberately free of torch so that EDA runs and their tests stay light.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FAMILIES: tuple[str, ...] = (
    "Category",
    "SubCategory",
    "HeelHeight",
    "Insole",
    "Closure",
    "Gender",
    "Material",
    "ToeStyle",
)
SOFTMAX_FAMILIES: tuple[str, ...] = ("Category", "SubCategory", "HeelHeight")
BCE_FAMILIES: tuple[str, ...] = tuple(f for f in FAMILIES if f not in SOFTMAX_FAMILIES)
IMAGE_SIZE: int = 136
PATH_DEPTH: int = 4


@dataclass(frozen=True)
class FamilySlice:
    name: str
    kind: str
    start: int
    end: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class LabelSchema:
    columns: tuple[str, ...]
    families: tuple[FamilySlice, ...]

    def family(self, name: str) -> FamilySlice:
        for fam in self.families:
            if fam.name == name:
                return fam
        raise KeyError(name)

    def softmax_families(self) -> tuple[FamilySlice, ...]:
        return tuple(f for f in self.families if f.kind == "softmax")

    def bce_families(self) -> tuple[FamilySlice, ...]:
        return tuple(f for f in self.families if f.kind == "bce")

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "families": [
                {
                    "name": f.name,
                    "kind": f.kind,
                    "start": f.start,
                    "end": f.end,
                    "labels": list(f.labels),
                }
                for f in self.families
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LabelSchema:
        families = tuple(
            FamilySlice(
                name=str(f["name"]),
                kind=str(f["kind"]),
                start=int(f["start"]),
                end=int(f["end"]),
                labels=tuple(str(x) for x in f["labels"]),
            )
            for f in payload["families"]  # type: ignore[union-attr]
        )
        return cls(columns=tuple(str(c) for c in payload["columns"]), families=families)  # type: ignore[union-attr]

    @classmethod
    def load(cls, path: Path) -> LabelSchema:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Catalog:
    frame: pd.DataFrame
    label_columns: tuple[str, ...]
    stats: dict[str, int]


def family_of(column: str) -> str:
    return column.split(".", 1)[0]


def family_kind(family: str) -> str:
    return "softmax" if family in SOFTMAX_FAMILIES else "bce"


def family_columns(columns: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for col in columns:
        fam = family_of(col)
        if fam in grouped:
            grouped[fam].append(col)
    return grouped


def cid_from_stem(stem: str) -> str | None:
    parts = stem.split(".")
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    return f"{parts[0]}-{parts[1]}"


def parse_image_paths(image_root: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for path in sorted(image_root.rglob("*.jpg")):
        relative = path.relative_to(image_root)
        if len(relative.parts) != PATH_DEPTH:
            continue
        cid = cid_from_stem(path.stem)
        if cid is None:
            continue
        product_id, color_id = cid.split("-")
        records.append(
            {
                "cid": cid,
                "product_id": product_id,
                "color_id": color_id,
                "dir_cat": relative.parts[0],
                "dir_sub": relative.parts[1],
                "brand": relative.parts[2],
                "path": relative.as_posix(),
            }
        )
    columns = ["cid", "product_id", "color_id", "dir_cat", "dir_sub", "brand", "path"]
    return pd.DataFrame.from_records(records, columns=columns)


def load_label_frame(bin_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(bin_csv, low_memory=False)
    frame["CID"] = frame["CID"].astype(str).str.strip()
    label_cols = [c for c in frame.columns if c != "CID"]
    frame[label_cols] = frame[label_cols].fillna(0).astype(np.int8)
    return frame


def build_catalog(image_root: Path, bin_csv: Path) -> Catalog:
    files = parse_image_paths(image_root)
    labels = load_label_frame(bin_csv)
    label_cols = tuple(c for c in labels.columns if c != "CID")

    n_files = len(files)
    ranked = files.assign(escapes=files["path"].str.count("%"))
    deduped = (
        ranked.sort_values(["escapes", "path"]).drop_duplicates("cid", keep="first").drop(columns="escapes")
    )
    n_duplicates = n_files - len(deduped)

    known = set(labels["CID"])
    orphans = int((~deduped["cid"].isin(known)).sum())

    frame = deduped.merge(labels, left_on="cid", right_on="CID", how="inner")
    frame = frame.drop(columns=["CID"]).sort_values("cid").reset_index(drop=True)

    stats = {
        "image_files": n_files,
        "metadata_rows": len(labels),
        "duplicate_cids": n_duplicates,
        "orphan_files": orphans,
        "joined_rows": len(frame),
    }
    return Catalog(frame=frame, label_columns=label_cols, stats=stats)


def label_matrix(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> np.ndarray:
    return frame.loc[:, list(columns)].to_numpy(dtype=np.int8)


def family_observed(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> np.ndarray:
    grouped = family_columns(columns)
    observed = np.zeros((len(frame), len(FAMILIES)), dtype=bool)
    for i, fam in enumerate(FAMILIES):
        cols = grouped[fam]
        if not cols:
            continue
        observed[:, i] = frame.loc[:, cols].to_numpy(dtype=np.int8).sum(axis=1) >= 1
    return observed


def positives_per_label(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    rows: np.ndarray | None = None,
) -> pd.Series:
    subset = frame if rows is None else frame.iloc[rows]
    counts = subset.loc[:, list(columns)].to_numpy(dtype=np.int32).sum(axis=0)
    return pd.Series(counts, index=list(columns))


def surviving_labels(counts: pd.Series, threshold: int) -> list[str]:
    return [str(c) for c in counts.index if counts[c] >= threshold]


def build_label_schema(surviving: list[str], all_columns: tuple[str, ...] | list[str]) -> LabelSchema:
    grouped = family_columns(all_columns)
    keep = set(surviving)
    ordered: list[str] = []
    families: list[FamilySlice] = []
    for fam in FAMILIES:
        cols = [c for c in grouped[fam] if c in keep]
        if not cols:
            continue
        start = len(ordered)
        ordered.extend(cols)
        families.append(
            FamilySlice(name=fam, kind=family_kind(fam), start=start, end=len(ordered), labels=tuple(cols))
        )
    return LabelSchema(columns=tuple(ordered), families=tuple(families))


def expand_family_mask(family_observed_matrix: np.ndarray, schema: LabelSchema) -> np.ndarray:
    expanded = np.zeros((family_observed_matrix.shape[0], len(schema.columns)), dtype=bool)
    for fam in schema.families:
        index = FAMILIES.index(fam.name)
        expanded[:, fam.start : fam.end] = family_observed_matrix[:, index : index + 1]
    return expanded


def softmax_targets(labels: np.ndarray, schema: LabelSchema) -> np.ndarray:
    families = schema.softmax_families()
    targets = np.full((labels.shape[0], len(families)), -1, dtype=np.int16)
    for j, fam in enumerate(families):
        block = labels[:, fam.start : fam.end]
        has_positive = block.sum(axis=1) >= 1
        targets[has_positive, j] = block[has_positive].argmax(axis=1).astype(np.int16)
    return targets


def normalize_image(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    rgb = image.convert("RGB")
    if max(rgb.size) > size:
        scale = size / max(rgb.size)
        resized = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        rgb = rgb.resize(resized, Image.Resampling.BICUBIC)
    if rgb.size == (size, size):
        return rgb
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(rgb, ((size - rgb.width) // 2, (size - rgb.height) // 2))
    return canvas
