"""CLIP zero-shot over the eight families, answering whether task-specific training is needed at all."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import data_setup, features, metrics, reporting
from src.catalog import LabelSchema
from src.glossary import Glossary
from src.reporting import PROCESSED_DIR, RUNS_ROOT, TEST_WITHHELD

RUN_NAME: str = "clip_zeroshot"
CLIP_MODEL: str = "ViT-B-32"
CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
EXCLUDED_FAMILY: str = "Insole"
BARE: str = "bare"
TEMPLATES: dict[str, str] = {
    BARE: "{phrase}",
    "photo": "a photo of {phrase}",
    "product": "a product photo of {phrase}",
    "catalog": "a product photo of {phrase}, on a plain white background",
}
SELECTED_ON: str = "map_scoreable on validation"

VAL_EXTRAS: frozenset[str] = frozenset(
    {
        "map_scoreable",
        "map_bce_scoreable",
        "not_zero_shot_scoreable",
        "zero_shot_excluded_family",
        "map_no_prompt_engineering",
        "map_scoreable_no_prompt_engineering",
        "macro_f1_basis",
    }
)

NOTE_MAP: str = (
    "map is computed on uncalibrated probabilities, as in every other run. Nothing was trained here, "
    "but the prompt template was chosen on validation, so map is threshold-free rather than wholly "
    "untouched by validation; map_no_prompt_engineering is the bare-template figure for comparison."
)
NOTE_MACRO_F1: str = (
    "thresholds are swept on validation, so macro_f1_bce is validation-calibrated rather than "
    "zero-shot; macro_f1_bce_out_of_fold refits per fold and is the honest one."
)


class ZeroShotError(RuntimeError):
    pass


def excluded_labels(schema: LabelSchema) -> list[str]:
    return list(schema.family(EXCLUDED_FAMILY).labels)


def scoreable_columns(schema: LabelSchema) -> list[int]:
    excluded = schema.family(EXCLUDED_FAMILY)
    return [c for c in range(len(schema.columns)) if not excluded.start <= c < excluded.end]


def assert_exclusions_match_glossary(glossary: Glossary, schema: LabelSchema) -> None:
    declared = sorted(glossary.excluded())
    family = sorted(excluded_labels(schema))
    if declared != family:
        difference = sorted(set(declared).symmetric_difference(family))
        raise ZeroShotError(
            "the glossary's zero_shot_scoreable flags and the excluded family have drifted apart. "
            f"The glossary marks {len(declared)} label(s) unscoreable and the {EXCLUDED_FAMILY} "
            f"family holds {len(family)}. Difference: {difference}"
        )


def build_clip(device: torch.device) -> tuple[object, object, dict]:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    preprocess = open_clip.get_model_preprocess_cfg(model)
    return model.to(device).eval(), tokenizer, preprocess


@torch.no_grad()
def encode_prompts(
    model: object, tokenizer: object, prompts: list[str], device: torch.device
) -> np.ndarray:
    tokens = tokenizer(prompts).to(device)  # type: ignore[operator]
    embedded = model.encode_text(tokens)  # type: ignore[attr-defined]
    return features.l2_normalize(embedded.float().cpu().numpy().astype(np.float64))


def rendered_prompts(glossary: Glossary, schema: LabelSchema, template: str) -> list[str]:
    return [template.format(phrase=glossary.prompt(name)) for name in schema.columns]


def apply_family_scaling(cosine: np.ndarray, schema: LabelSchema, logit_scale: float) -> np.ndarray:
    scores = np.array(cosine, dtype=np.float64, copy=True)
    for family in schema.softmax_families():
        scores[:, family.start : family.end] *= logit_scale
    return scores


def group_map(per_label: np.ndarray, columns: list[int]) -> float:
    return metrics.macro_average(per_label[columns])[0]


def template_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    family_observed: np.ndarray,
    schema: LabelSchema,
) -> dict[str, float]:
    probs = metrics.logits_to_probabilities(scores, schema)
    metrics.assert_bce_ranking_survives(scores, probs, schema)
    mask = metrics.cell_mask(family_observed, schema)
    per_label = metrics.per_label_average_precision(probs, labels, mask)

    softmax_columns = [c for f in schema.softmax_families() for c in range(f.start, f.end)]
    bce_columns = [c for f in schema.bce_families() for c in range(f.start, f.end)]
    scoreable = scoreable_columns(schema)
    return {
        "map": metrics.macro_average(per_label)[0],
        "map_scoreable": group_map(per_label, scoreable),
        "map_softmax": group_map(per_label, softmax_columns),
        "map_bce": group_map(per_label, bce_columns),
        "map_bce_scoreable": group_map(per_label, sorted(set(bce_columns).intersection(scoreable))),
    }


def caveats_from_glossary(glossary: Glossary, schema: LabelSchema) -> dict[str, str]:
    excluded = set(excluded_labels(schema))
    return {
        name: glossary.entries[name].notes
        for name in schema.columns
        if name not in excluded and glossary.entries[name].notes
    }


def zero_shot_scores(
    base: dict[str, object],
    searched: dict[str, dict[str, object]],
    selected: str,
    schema: LabelSchema,
) -> dict[str, object]:
    scores = dict(base)
    scores.update(
        {
            "map_scoreable": searched[selected]["map_scoreable"],
            "map_bce_scoreable": searched[selected]["map_bce_scoreable"],
            "not_zero_shot_scoreable": excluded_labels(schema),
            "zero_shot_excluded_family": EXCLUDED_FAMILY,
            "map_no_prompt_engineering": searched[BARE]["map"],
            "map_scoreable_no_prompt_engineering": searched[BARE]["map_scoreable"],
            "macro_f1_basis": "validation-calibrated",
        }
    )
    return scores


def build_report(
    scores: dict[str, object],
    searched: dict[str, dict[str, object]],
    selected: str,
    glossary: Glossary,
    schema: LabelSchema,
    data_cfg: data_setup.DataConfig,
    logit_scale: float,
    image_size: int,
) -> dict[str, object]:
    return {
        "run": RUN_NAME,
        "stem": RUN_NAME,
        "role": "candidate",
        "checkpoint": None,
        "epoch": None,
        "selection_metric": None,
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAINED,
        "clip_logit_scale": logit_scale,
        "mean": list(data_cfg.mean),
        "std": list(data_cfg.std),
        "image_size": image_size,
        "template_search": {
            "selected": selected,
            "selected_on": SELECTED_ON,
            "bare_template": BARE,
            "templates": searched,
        },
        "caveats": caveats_from_glossary(glossary, schema),
        "val": scores,
        "test": TEST_WITHHELD,
    }


def run(
    processed_dir: Path = PROCESSED_DIR,
    runs_root: Path = RUNS_ROOT,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    device: torch.device | None = None,
) -> dict[str, object]:
    artifacts = data_setup.load_artifacts(processed_dir)
    schema = artifacts.schema
    glossary = Glossary.load()
    assert_exclusions_match_glossary(glossary, schema)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, preprocess = build_clip(device)
    logit_scale = float(model.logit_scale.exp().detach())  # type: ignore[attr-defined]

    data_cfg = data_setup.DataConfig(
        processed_dir=processed_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mean=tuple(preprocess["mean"]),
        std=tuple(preprocess["std"]),
    )
    rows = artifacts.splits["val"]
    loader = data_setup.build_dataloader(data_cfg, artifacts, "val")
    encoded = features.extract_features(model.encode_image, loader, device)  # type: ignore[attr-defined]
    image_features = features.l2_normalize(encoded.astype(np.float64))

    labels = artifacts.labels[rows].astype(np.float64)
    family_observed = artifacts.family_observed[rows]

    searched: dict[str, dict[str, object]] = {}
    cached: dict[str, np.ndarray] = {}
    for name, template in TEMPLATES.items():
        prompts = rendered_prompts(glossary, schema, template)
        text = encode_prompts(model, tokenizer, prompts, device)
        scored = apply_family_scaling(image_features @ text.T, schema, logit_scale)
        cached[name] = scored
        searched[name] = {
            "template": template,
            **template_scores(scored, labels, family_observed, schema),
        }

    selected = max(searched, key=lambda name: float(searched[name]["map_scoreable"]))
    predictions = reporting.Predictions(
        logits=cached[selected], labels=labels, family_observed=family_observed, rows=rows
    )
    base, calibrated, thresholds, calibrators = reporting.score_split(
        predictions, schema, note_map=NOTE_MAP, note_macro_f1=NOTE_MACRO_F1
    )
    scores = zero_shot_scores(base, searched, selected, schema)
    reporting.assert_contract(scores, RUN_NAME, extra=VAL_EXTRAS)

    run_dir = runs_root / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        scores, searched, selected, glossary, schema, data_cfg, logit_scale, image_size
    )
    reporting.write_report(run_dir, report)
    reporting.write_confusion(run_dir, calibrated, predictions, schema)
    np.save(run_dir / reporting.PROBS_NAME.format(split="val"), calibrated.astype(np.float32))
    (run_dir / reporting.THRESHOLDS_NAME).write_text(
        json.dumps({name: float(thresholds[i]) for i, name in enumerate(schema.columns)}, indent=2),
        encoding="utf-8",
    )
    (run_dir / reporting.CALIBRATION_NAME).write_text(
        json.dumps(calibrators.to_dict(), indent=2), encoding="utf-8"
    )
    reporting.write_summary(
        run_dir,
        {
            "name": RUN_NAME,
            "stem": RUN_NAME,
            "role": "candidate",
            "selection_metric": None,
            "selection_scope": "no epoch is selected; the prompt template is chosen on validation",
            "seed": None,
            "epochs_completed": 0,
            "device": str(device),
            "clip_model": CLIP_MODEL,
            "clip_pretrained": CLIP_PRETRAINED,
            "template_selected": selected,
            "note": (
                "zero-shot CLIP scored through the shared contract; thresholds are validation "
                "calibrated and the excluded Insole family is reported separately"
            ),
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CLIP zero-shot against the catalog attributes")
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    report = run(
        processed_dir=args.processed,
        runs_root=args.runs_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=torch.device("cpu") if args.cpu else None,
    )
    val = report["val"]
    search = report["template_search"]
    print(f"{CLIP_MODEL} / {CLIP_PRETRAINED}   template {search['selected']!r} on {SELECTED_ON}")
    for name, row in search["templates"].items():
        print(f"  {name:9s} map {row['map']:.4f}   scoreable {row['map_scoreable']:.4f}")
    print(f"mAP            {val['map']:.4f} over 107   {val['map_scoreable']:.4f} over 94 scoreable")
    print(
        f"  bare prompt  {val['map_no_prompt_engineering']:.4f} over 107   "
        f"{val['map_scoreable_no_prompt_engineering']:.4f} over 94"
    )
    print(
        f"  softmax {val['map_softmax']:.4f}   bce {val['map_bce']:.4f} "
        f"(scoreable {val['map_bce_scoreable']:.4f})"
    )
    print(
        f"macro F1 bce   {val['macro_f1_bce']:.4f} validation-calibrated   "
        f"out-of-fold {val['macro_f1_bce_out_of_fold']:.4f}"
    )
    print(f"excluded       {len(val['not_zero_shot_scoreable'])} {EXCLUDED_FAMILY} labels")
    for name, note in report["caveats"].items():
        print(f"  caveat {name}: {note}")


if __name__ == "__main__":
    main()
