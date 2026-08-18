"""Runs a frozen encoder over a split once and caches the result, so repeated passes stay cheap."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

CACHE_DIRNAME: str = "features"
Encoder = Callable[[Tensor], Tensor]
Progress = Callable[[Iterable[object]], Iterable[object]]


def cache_path(root: Path, tag: str, image_size: int, split: str) -> Path:
    return Path(root) / CACHE_DIRNAME / f"{tag}_{image_size}_{split}.npy"


@torch.no_grad()
def extract_features(
    encode: Encoder,
    loader: Iterable[tuple[Tensor, ...]],
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    progress: Progress | None = None,
) -> np.ndarray:
    stream = loader if progress is None else progress(loader)
    chunks: list[Tensor] = []
    for batch in stream:
        images = batch[0].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"
        ):
            chunks.append(encode(images).float().cpu())
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(chunks).numpy().astype(np.float32)


def load_or_extract(
    path: Path,
    extract: Callable[[], np.ndarray],
    rows: int | None = None,
) -> np.ndarray:
    if path.exists():
        cached = np.load(path)
        if rows is not None and cached.shape[0] != rows:
            raise ValueError(
                f"{path} holds {cached.shape[0]} rows but the split has {rows}; the cache is stale "
                "and reusing it would score one split's features against another's labels"
            )
        return cached
    features = extract()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, features)
    temporary.replace(path)
    return features


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(features.dtype).tiny)
