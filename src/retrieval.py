"""Same-product retrieval over the whole catalog, from three embedding sources so fine-tuning is
separated from pretraining and architecture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import catalog, data_setup, evaluate, features, reporting
from src.catalog import LabelSchema

EMBEDDINGS_NAME: str = "embeddings_{source}.npy"
TIMING_NAME: str = "retrieval_timing.json"
INDEX_NOTE: str = (
    "The retrieval index covers the whole catalog, training images included. That is intended for the "
    "demo and it means the neighbours are not a held-out result; the README must say so."
)


class RetrievalError(RuntimeError):
    pass


def catalog_product_ids(processed_dir: Path) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(processed_dir / "catalog.csv")
    return frame["product_id"].to_numpy()


def all_rows_split(artifacts: data_setup.Artifacts) -> data_setup.Artifacts:
    total = artifacts.labels.shape[0]
    splits = dict(artifacts.splits)
    splits["all"] = np.arange(total, dtype=np.int64)
    return data_setup.Artifacts(
        schema=artifacts.schema,
        labels=artifacts.labels,
        family_observed=artifacts.family_observed,
        splits=splits,
        manifest=artifacts.manifest,
    )


def encoder_for(source: str, run_dir: Path, device: torch.device) -> tuple[features.Encoder, dict]:
    if source not in features.SOURCES:
        raise RetrievalError(
            f"unknown embedding source {source!r}, expected one of {features.SOURCES}"
        )
    if source == features.SOURCE_CLIP:
        return features.clip_image_encoder(device)
    payload = evaluate.load_checkpoint(run_dir)
    backbone = str(dict(payload["config"])["backbone"])
    if source == features.SOURCE_PRETRAINED:
        return features.timm_image_encoder(backbone, device)
    return features.timm_image_encoder(backbone, device, state_dict=payload["model"])


def embed_catalog(
    source: str,
    run_dir: Path,
    processed_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[np.ndarray, dict]:
    artifacts = all_rows_split(data_setup.load_artifacts(processed_dir))
    encode, meta = encoder_for(source, run_dir, device)
    cfg = data_setup.DataConfig(
        processed_dir=processed_dir,
        image_size=meta["image_size"],
        batch_size=batch_size,
        num_workers=num_workers,
        mean=meta["mean"],
        std=meta["std"],
    )
    loader = data_setup.build_eval_dataloader(cfg, artifacts, "all")
    raw = features.extract_features(encode, loader, device)
    embeddings = features.l2_normalize(raw)
    features.assert_embeddings_usable(embeddings, artifacts.labels.shape[0], source)
    meta["rows"] = int(embeddings.shape[0])
    meta["dim"] = int(embeddings.shape[1])
    return embeddings, meta


def brute_force_neighbours(
    embeddings: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    scores = embeddings[queries] @ embeddings.T
    scores[np.arange(len(queries)), queries] = -np.inf
    top = np.argpartition(-scores, kth=k, axis=1)[:, :k]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), axis=1), 1)
    return ordered, np.take_along_axis(scores, ordered, 1)


def time_brute_force(embeddings: np.ndarray, queries: np.ndarray, k: int, repeats: int) -> dict:
    sample = queries[: min(len(queries), 256)]
    brute_force_neighbours(embeddings, sample[:8], k)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        brute_force_neighbours(embeddings, sample, k)
        elapsed.append(time.perf_counter() - start)
    per_query = float(np.median(elapsed) / len(sample))
    return {
        "gallery_vectors": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "queries_timed": int(len(sample)),
        "repeats": repeats,
        "median_batch_seconds": float(np.median(elapsed)),
        "seconds_per_query": per_query,
        "projected_seconds_for_full_query_set": per_query * int(len(queries)),
        "note": (
            "brute force over the whole catalog, measured before considering an index. The standing "
            "rule is to measure before adding Faiss."
        ),
    }


def run(
    run_dir: Path,
    processed_dir: Path,
    sources: tuple[str, ...],
    k: int,
    repeats: int,
    device: torch.device,
) -> dict:
    schema = LabelSchema.load(processed_dir / "label_schema.json")
    product_ids = catalog_product_ids(processed_dir)
    query_set = catalog.retrieval_query_set(product_ids)
    report: dict[str, object] = {
        "run": run_dir.name,
        "gallery": "whole catalog including training images",
        "index_note": INDEX_NOTE,
        "labels": len(schema.columns),
        "query_set": {
            "products": query_set.products,
            "images": query_set.images,
            "excluded_products": query_set.excluded_products,
            "excluded_images": query_set.excluded_images,
        },
        "sources": {},
    }
    for source in sources:
        embeddings, meta = embed_catalog(source, run_dir, processed_dir, device)
        path = run_dir / EMBEDDINGS_NAME.format(source=source)
        np.save(path, embeddings.astype(np.float32))
        meta["path"] = path.name
        meta["timing"] = time_brute_force(embeddings, query_set.rows, k, repeats)
        report["sources"][source] = meta
        print(f"{source:<12} {embeddings.shape}  ->  {path.name}  "
              f"{meta['timing']['seconds_per_query'] * 1000:.2f} ms/query")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract retrieval embeddings and time brute force")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=reporting.PROCESSED_DIR)
    parser.add_argument("--sources", nargs="+", default=list(features.SOURCES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    report = run(args.run, args.processed, tuple(args.sources), args.k, args.repeats, device)
    destination = args.run / TIMING_NAME
    destination.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten to {destination}")
    print(INDEX_NOTE)


if __name__ == "__main__":
    main()
