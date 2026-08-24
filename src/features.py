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


CLIP_MODEL: str = "ViT-B-32"
CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
SOURCE_FINETUNED: str = "finetuned"
SOURCE_PRETRAINED: str = "pretrained"
SOURCE_CLIP: str = "clip"
SOURCES: tuple[str, ...] = (SOURCE_FINETUNED, SOURCE_PRETRAINED, SOURCE_CLIP)


class FeatureError(RuntimeError):
    pass


def build_clip(device: torch.device) -> tuple[object, object, dict]:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    preprocess = open_clip.get_model_preprocess_cfg(model)
    return model.to(device).eval(), tokenizer, preprocess


def clip_image_encoder(device: torch.device) -> tuple[Encoder, dict]:
    model, _tokenizer, preprocess = build_clip(device)

    def encode(images: Tensor) -> Tensor:
        return model.encode_image(images)

    return encode, {
        "source": SOURCE_CLIP,
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAINED,
        "mean": tuple(preprocess["mean"]),
        "std": tuple(preprocess["std"]),
        "image_size": int(preprocess["size"][-1]),
    }


def timm_image_encoder(
    backbone: str, device: torch.device, state_dict: dict | None = None
) -> tuple[Encoder, dict]:
    import timm
    from timm.data import resolve_model_data_config

    model = timm.create_model(backbone, pretrained=state_dict is None, num_classes=0)
    if state_dict is not None:
        missing = model.load_state_dict(
            {k: v for k, v in state_dict.items() if not k.startswith("head.fc.")}, strict=False
        )
        if missing.unexpected_keys:
            raise FeatureError(
                f"{len(missing.unexpected_keys)} unexpected keys loading {backbone}: "
                f"{missing.unexpected_keys[:4]}. The checkpoint does not match this backbone."
            )
    model = model.to(device).eval()
    cfg = resolve_model_data_config(model)

    def encode(images: Tensor) -> Tensor:
        return model(images)

    return encode, {
        "source": SOURCE_PRETRAINED if state_dict is None else SOURCE_FINETUNED,
        "backbone": backbone,
        "mean": tuple(cfg["mean"]),
        "std": tuple(cfg["std"]),
        "image_size": int(cfg["input_size"][-1]),
    }


def assert_embeddings_usable(features: np.ndarray, rows: int, source: str) -> None:
    if features.shape[0] != rows:
        raise FeatureError(
            f"{source} produced {features.shape[0]} embeddings for {rows} catalog rows; retrieval "
            "indexes by row position, so a mismatch would rank the wrong images"
        )
    if not np.isfinite(features).all():
        raise FeatureError(f"{source} embeddings contain non-finite values")
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise FeatureError(
            f"{source} embeddings are not L2 normalised (norms {norms.min():.4f}..{norms.max():.4f}); "
            "cosine similarity by dot product assumes they are"
        )
