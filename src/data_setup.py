"""The memmap-backed Dataset and the dataloaders. Opens the image array lazily so workers stay cheap."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src import corruption
from src.catalog import LabelSchema
from src.utils import seed_worker

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGES_NAME: str = "images_u8.npy"
LABELS_NAME: str = "labels.npy"
OBSERVED_NAME: str = "family_observed.npy"
SCHEMA_NAME: str = "label_schema.json"
SPLITS_NAME: str = "splits.json"
MANIFEST_NAME: str = "manifest.json"
DEFAULT_CROP_RATIO: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)


@dataclass(frozen=True)
class DataConfig:
    processed_dir: Path
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    crop_scale: tuple[float, float] = (0.8, 1.0)
    crop_ratio: tuple[float, float] = DEFAULT_CROP_RATIO
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


@dataclass(frozen=True)
class Artifacts:
    schema: LabelSchema
    labels: np.ndarray
    family_observed: np.ndarray
    splits: dict[str, np.ndarray]
    manifest: dict = field(default_factory=dict)
    corruption: dict = field(default_factory=dict)
    corruption_mask: np.ndarray | None = None


def load_artifacts(
    processed_dir: Path,
    corrupt_rate: float = 0.0,
    corrupt_seed: int = 42,
    corrupt_type: str = corruption.TYPE_UNIFORM,
) -> Artifacts:
    schema = LabelSchema.load(processed_dir / SCHEMA_NAME)
    labels = np.load(processed_dir / LABELS_NAME)
    observed = np.load(processed_dir / OBSERVED_NAME)
    planted: dict = {}
    mask = None
    if corrupt_rate > 0.0:
        source = corruption.corruption_dir(processed_dir, corrupt_rate, corrupt_seed, corrupt_type)
        corrupted, mask, planted = corruption.load_corruption(source)
        corruption.assert_type_matches(planted, corrupt_type, source)
        corruption.assert_shapes_match(labels, corrupted, mask)
        labels = corrupted
    payload = json.loads((processed_dir / SPLITS_NAME).read_text(encoding="utf-8"))
    splits = {name: np.asarray(idx, dtype=np.int64) for name, idx in payload["indices"].items()}
    manifest_path = processed_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if labels.shape[1] != len(schema.columns):
        raise ValueError(f"labels has {labels.shape[1]} columns, schema declares {len(schema.columns)}")
    if labels.shape[0] != observed.shape[0]:
        raise ValueError(f"labels has {labels.shape[0]} rows, family_observed has {observed.shape[0]}")
    return Artifacts(
        schema=schema,
        labels=labels,
        family_observed=observed,
        splits=splits,
        manifest=manifest,
        corruption=planted,
        corruption_mask=mask,
    )


def build_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    train: bool,
    crop_scale: tuple[float, float] = (0.8, 1.0),
    crop_ratio: tuple[float, float] = DEFAULT_CROP_RATIO,
) -> Callable[[torch.Tensor], torch.Tensor]:
    geometry: list[Callable[[torch.Tensor], torch.Tensor]] = (
        [
            v2.RandomResizedCrop(
                image_size, scale=crop_scale, ratio=crop_ratio,
                interpolation=v2.InterpolationMode.BICUBIC, antialias=True
            ),
            v2.RandomHorizontalFlip(),
        ]
        if train
        else [
            v2.Resize(
                (image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True
            )
        ]
    )
    return v2.Compose([*geometry, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])


class ShoeDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        family_observed: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.images_path = Path(processed_dir) / IMAGES_NAME
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.family_observed = family_observed
        self.transform = transform
        self._images: np.ndarray | None = None

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        return self._images

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = int(self.indices[position])
        pixels = torch.from_numpy(np.asarray(self.images[row]).copy()).permute(2, 0, 1)
        image = self.transform(pixels)
        labels = torch.from_numpy(self.labels[row].astype(np.float32))
        observed = torch.from_numpy(self.family_observed[row].astype(np.bool_))
        return image, labels, observed


def build_dataset(cfg: DataConfig, artifacts: Artifacts, split: str) -> ShoeDataset:
    transform = build_transforms(
        cfg.image_size, cfg.mean, cfg.std, train=(split == "train"),
        crop_scale=cfg.crop_scale, crop_ratio=cfg.crop_ratio,
    )
    return ShoeDataset(
        processed_dir=cfg.processed_dir,
        indices=artifacts.splits[split],
        labels=artifacts.labels,
        family_observed=artifacts.family_observed,
        transform=transform,
    )


def build_dataloader(cfg: DataConfig, artifacts: Artifacts, split: str) -> DataLoader:
    dataset = build_dataset(cfg, artifacts, split)
    is_train = split == "train"
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_eval_dataloader(cfg: DataConfig, artifacts: Artifacts, split: str) -> DataLoader:
    transform = build_transforms(cfg.image_size, cfg.mean, cfg.std, train=False)
    dataset = ShoeDataset(
        processed_dir=cfg.processed_dir,
        indices=artifacts.splits[split],
        labels=artifacts.labels,
        family_observed=artifacts.family_observed,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
    )


def build_dataloaders(
    cfg: DataConfig, artifacts: Artifacts, splits: tuple[str, ...] = ("train", "val")
) -> dict[str, DataLoader]:
    return {split: build_dataloader(cfg, artifacts, split) for split in splits}
