"""Scores persisted retrieval embeddings against the definitions fixed in FINDINGS 25, and reports the
same numbers by query provenance so the catalog-wide figure can be read split by split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import catalog, metrics

SCORED_NAME: str = "retrieval_scored.json"
EMBEDDINGS_NAME: str = "embeddings_{source}.npy"
DEFAULT_SOURCES: tuple[str, ...] = ("finetuned", "pretrained", "clip")
DEFAULT_K: int = 10
BOOTSTRAP_RESAMPLES: int = 2000
BOOTSTRAP_SEED: int = 20260824
BOOTSTRAP_LEVEL: float = 0.95
QUERY_BATCH: int = 256
NOISE_FLOOR: float = 0.001
STRATA: tuple[str, ...] = ("train", "val", "test")
KNOWN_TIE_ORDER_NOTE: str = (
    "The figures in retrieval.json were produced by a script that was never committed, and it ordered "
    "tied similarities with numpy's default unstable sort. This entry point uses metrics.ranked_relevance, "
    "which sorts stably and breaks ties by ascending index. Both orderings were measured against the "
    "published value: the stable one lands 1.29e-06 away and the unstable one reproduces it exactly. Only "
    "the two order-sensitive fields of the fine-tuned source move, because that space clusters colorway "
    "siblings tightly enough to produce exact ties; recall@10, cmc@1 and cmc@10 are bit-identical, so the "
    "top-10 membership agrees and only the order within it differs. The stable ordering is kept because an "
    "unstable sort's tie order can change with the numpy version, and because picking the rule that "
    "matches a target after seeing which one does is the move this project refuses elsewhere."
)
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("C1_finetuned_vs_pretrained", "finetuned", "pretrained"),
    ("C2_finetuned_vs_clip", "finetuned", "clip"),
    ("C3_pretrained_vs_clip", "pretrained", "clip"),
)


class RetrievalScoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerQuery:
    average_precision: np.ndarray
    recall_capped: np.ndarray
    recall_uncapped: np.ndarray
    cmc_at_1: np.ndarray
    cmc_at_10: np.ndarray
    relevant: np.ndarray


def catalog_product_ids(processed_dir: Path) -> np.ndarray:
    return pd.read_csv(processed_dir / "catalog.csv")["product_id"].to_numpy()


def load_embeddings(run_dir: Path, source: str) -> np.ndarray:
    path = run_dir / EMBEDDINGS_NAME.format(source=source)
    if not path.exists():
        raise RetrievalScoreError(
            f"{path.name} is missing. Extract the embeddings with `python -m src.retrieval --run "
            f"{run_dir}` before scoring them."
        )
    return np.load(path).astype(np.float32)


def query_strata(processed_dir: Path, rows: np.ndarray, total: int) -> np.ndarray:
    indices = json.loads((processed_dir / "splits.json").read_text(encoding="utf-8"))["indices"]
    owner = np.full(total, "", dtype=object)
    for name in STRATA:
        owner[np.asarray(indices[name], dtype=np.int64)] = name
    assigned = owner[np.asarray(rows, dtype=np.int64)]
    if any(value == "" for value in assigned):
        raise RetrievalScoreError(
            "a query row belongs to no split, so the provenance stratification would silently drop it. "
            "splits.json does not cover every catalog row."
        )
    return assigned.astype(str)


def score_source(
    embeddings: np.ndarray, product_ids: np.ndarray, queries: np.ndarray, k: int
) -> PerQuery:
    ids = np.asarray(product_ids)
    rows = np.asarray(queries, dtype=np.int64)
    average_precision = np.empty(rows.size, dtype=np.float64)
    recall_capped = np.empty(rows.size, dtype=np.float64)
    recall_uncapped = np.empty(rows.size, dtype=np.float64)
    cmc_at_1 = np.empty(rows.size, dtype=np.float64)
    cmc_at_10 = np.empty(rows.size, dtype=np.float64)
    relevant_counts = np.empty(rows.size, dtype=np.int64)

    for start in range(0, rows.size, QUERY_BATCH):
        block = rows[start : start + QUERY_BATCH]
        similarity = (embeddings[block] @ embeddings.T).astype(np.float64)
        for offset, query in enumerate(block):
            relevant = ids == ids[query]
            total = int(relevant.sum()) - 1
            hits = metrics.ranked_relevance(similarity[offset], relevant, int(query))
            position = start + offset
            relevant_counts[position] = total
            average_precision[position] = metrics.average_precision_at_rank(hits[:k], total)
            recall_capped[position] = metrics.recall_at_k(hits, total, k, metrics.CAPPED)
            recall_uncapped[position] = metrics.recall_at_k(hits, total, k, metrics.UNCAPPED)
            cmc_at_1[position] = metrics.cmc_at_k(hits, 1)
            cmc_at_10[position] = metrics.cmc_at_k(hits, k)

    return PerQuery(
        average_precision=average_precision,
        recall_capped=recall_capped,
        recall_uncapped=recall_uncapped,
        cmc_at_1=cmc_at_1,
        cmc_at_10=cmc_at_10,
        relevant=relevant_counts,
    )


def macro_over_products(values: np.ndarray, query_products: np.ndarray) -> float:
    frame = pd.DataFrame({"product": np.asarray(query_products), "value": np.asarray(values)})
    return float(frame.groupby("product", sort=True)["value"].mean().mean())


def summarise(scored: PerQuery, query_products: np.ndarray, mask: np.ndarray | None = None) -> dict:
    keep = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    counts = scored.relevant[keep].astype(np.float64)
    return {
        "queries": int(counts.size),
        "map@10": float(np.nanmean(scored.average_precision[keep])),
        "map@10_macro_over_products": macro_over_products(
            scored.average_precision[keep], np.asarray(query_products)[keep]
        ),
        "recall@10_capped": float(np.nanmean(scored.recall_capped[keep])),
        "recall@10_uncapped": float(np.nanmean(scored.recall_uncapped[keep])),
        "cmc@1": float(np.nanmean(scored.cmc_at_1[keep])),
        "cmc@10": float(np.nanmean(scored.cmc_at_10[keep])),
        "relevant_per_query": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "queries_with_one": int((counts == 1).sum()),
            "share_with_one": float((counts == 1).mean()),
        },
    }


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> dict:
    paired = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picks = generator.integers(0, paired.size, paired.size)
        draws[index] = paired[picks].mean()
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "delta": float(paired.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def compare_published(fresh: dict, published_path: Path) -> dict:
    if not published_path.exists():
        return {"published_file": published_path.name, "status": "absent, nothing to compare"}
    published = json.loads(published_path.read_text(encoding="utf-8"))
    differences: dict[str, dict] = {}
    for source, scores in published.get("sources", {}).items():
        mine = fresh.get(source, {})
        for key, value in scores.items():
            if not isinstance(value, (int, float)) or key not in mine:
                continue
            delta = float(mine[key]) - float(value)
            if abs(delta) > 1e-9:
                differences[f"{source}/{key}"] = {
                    "published": float(value), "recomputed": float(mine[key]), "delta": delta
                }
    largest = max((abs(d["delta"]) for d in differences.values()), default=0.0)
    if not differences:
        status = "reproduced"
    elif largest < NOISE_FLOOR:
        status = f"differs below the {NOISE_FLOOR} noise floor, cause identified"
    else:
        status = "DIFFERS materially, do not publish either until resolved"
    return {
        "published_file": published_path.name,
        "fields_that_moved": len(differences),
        "largest_absolute_delta": largest,
        "noise_floor": NOISE_FLOOR,
        "differences": differences,
        "identified_cause": KNOWN_TIE_ORDER_NOTE if differences else "",
        "status": status,
    }


def run(run_dir: Path, processed_dir: Path, sources: tuple[str, ...], k: int) -> dict:
    product_ids = catalog_product_ids(processed_dir)
    query_set = catalog.retrieval_query_set(product_ids)
    catalog.assert_query_set_matches_findings(query_set)
    strata = query_strata(processed_dir, query_set.rows, total=int(product_ids.size))

    scored: dict[str, PerQuery] = {}
    pooled: dict[str, dict] = {}
    for source in sources:
        embeddings = load_embeddings(run_dir, source)
        if embeddings.shape[0] != product_ids.size:
            raise RetrievalScoreError(
                f"{source} has {embeddings.shape[0]} vectors against {product_ids.size} catalog rows; "
                "the gallery and the ground truth are not the same set."
            )
        result = score_source(embeddings, product_ids, query_set.rows, k)
        scored[source] = result
        pooled[source] = {"dim": int(embeddings.shape[1]), **summarise(result, query_set.product_ids)}
        print(f"{source:<12} map@{k} {pooled[source]['map@10']:.4f}")

    by_stratum: dict[str, dict] = {}
    for name in STRATA:
        mask = strata == name
        by_stratum[name] = {
            source: summarise(result, query_set.product_ids, mask) for source, result in scored.items()
        }

    comparisons: dict[str, dict] = {}
    for label, left, right in COMPARISONS:
        if left not in scored or right not in scored:
            continue
        entry = {
            "pooled": paired_bootstrap(
                scored[left].average_precision, scored[right].average_precision
            )
        }
        for name in STRATA:
            mask = strata == name
            entry[name] = paired_bootstrap(
                scored[left].average_precision[mask], scored[right].average_precision[mask]
            )
        comparisons[label] = entry

    return {
        "WHAT_THIS_IS": (
            "retrieval on the FINDINGS 4.5 query set, scored against FINDINGS 25, recomputed by a "
            "committed entry point so every published number is reproducible from the repository"
        ),
        "run": run_dir.name,
        "primary_metric": f"map@{k} macro over queries",
        "k": k,
        "gallery": "whole catalog including training images",
        "index_note": (
            "The gallery is the WHOLE CATALOG including training images. The neighbours are not a "
            "held-out result and the README must say so."
        ),
        "ground_truth_note": (
            "ProductID ground truth measures same-product matching, not general visual similarity. "
            "Colour is not one of the eight families, so no colour-sensitive evaluation is possible."
        ),
        "query_set": {
            "products": query_set.products,
            "images": query_set.images,
            "excluded_products": query_set.excluded_products,
            "excluded_images": query_set.excluded_images,
            "excluded_reason": (
                "single-colorway products have no relevant item to find (FINDINGS 4.5)"
            ),
        },
        "queries_by_stratum": {name: int((strata == name).sum()) for name in STRATA},
        "sources": pooled,
        "by_stratum": by_stratum,
        "comparisons": comparisons,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "paired": True,
            "level": BOOTSTRAP_LEVEL,
            "generator": "numpy default_rng",
        },
        "reproduction_check": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score persisted retrieval embeddings")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run(args.run, args.processed, tuple(args.sources), args.k)
    report["reproduction_check"] = compare_published(
        report["sources"], args.run / "retrieval.json"
    )
    destination = args.out or (args.run / SCORED_NAME)
    destination.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"scored retrieval written to {destination}")
    print(f"reproduction against retrieval.json: {report['reproduction_check']['status']}")


if __name__ == "__main__":
    main()
