"""The Gradio demo: attribute prediction, catalog audit and same-product neighbours.

Two modes because an audit needs a stored value to contradict. Upload mode has none, so it shows
predictions and neighbours only; catalog mode compares the stored row against the prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import data_setup, evaluate, features, i18n, metrics, reporting
from src.catalog import FAMILIES, LabelSchema
from src.glossary import Glossary, assert_covers, assert_covers_families, assert_turkish_complete
from src.i18n import DEFAULT_LANGUAGE, LANGUAGE_NAMES, LANGUAGES, Locale

WINNER_RUN: Path = reporting.RUNS_ROOT / "convnext_base_ar1_s42"
EMBEDDINGS_NAME: str = "embeddings_finetuned.npy"
NEIGHBOURS: int = 10
CATALOG_CHOICES: int = 400

STATUS_CONFLICT: str = "conflict"
STATUS_FILL: str = "fill"
STATUS_AGREES: str = "agrees"


class AppError(RuntimeError):
    pass


@dataclass
class Bundle:
    model: torch.nn.Module
    schema: LabelSchema
    glossary: Glossary
    calibrators: metrics.Calibrators
    thresholds: np.ndarray
    embeddings: np.ndarray
    frame: pd.DataFrame
    labels: np.ndarray
    family_observed: np.ndarray
    images: np.ndarray
    device: torch.device
    transform: object
    run_name: str


def family_of(label: str) -> str:
    return label.split(".", 1)[0]


def calibrated_probabilities(bundle: Bundle, logits: np.ndarray) -> np.ndarray:
    return metrics.apply_calibrators(logits, bundle.schema, bundle.calibrators)


def as_batch(bundle: Bundle, image: np.ndarray) -> torch.Tensor:
    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = bundle.transform(torch.from_numpy(array).permute(2, 0, 1))
    return tensor.unsqueeze(0).to(bundle.device)


def predict_row(bundle: Bundle, image: np.ndarray) -> np.ndarray:
    batch = as_batch(bundle, image)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bundle.device.type == "cuda"):
        logits = bundle.model(batch).float().cpu().numpy().astype(np.float64)
    return calibrated_probabilities(bundle, logits)[0]


def family_predictions(bundle: Bundle, probabilities: np.ndarray) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for family in bundle.schema.families:
        block = probabilities[family.start : family.end]
        if family.kind == "softmax":
            index = int(block.argmax())
            grouped[family.name] = [(family.labels[index], float(block[index]))]
            continue
        picks = [
            (label, float(block[i]))
            for i, label in enumerate(family.labels)
            if block[i] >= bundle.thresholds[family.start + i]
        ]
        grouped[family.name] = sorted(picks, key=lambda item: -item[1])
    return grouped


def stored_values(bundle: Bundle, row: int) -> dict[str, list[str] | None]:
    stored: dict[str, list[str] | None] = {}
    for family in bundle.schema.families:
        if not bool(bundle.family_observed[row, FAMILIES.index(family.name)]):
            stored[family.name] = None
            continue
        block = bundle.labels[row, family.start : family.end]
        stored[family.name] = [family.labels[i] for i in np.flatnonzero(block > 0.5)]
    return stored


def audit_family(stored: list[str] | None, predicted: list[tuple[str, float]]) -> str:
    if stored is None:
        return STATUS_FILL if predicted else STATUS_AGREES
    if set(stored) == {label for label, _ in predicted}:
        return STATUS_AGREES
    return STATUS_CONFLICT


def audit_row(bundle: Bundle, row: int, probabilities: np.ndarray) -> dict[str, dict[str, object]]:
    predicted = family_predictions(bundle, probabilities)
    stored = stored_values(bundle, row)
    return {
        family.name: {
            "status": audit_family(stored[family.name], predicted[family.name]),
            "stored": stored[family.name],
            "predicted": predicted[family.name],
        }
        for family in bundle.schema.families
    }


def neighbours_for(bundle: Bundle, vector: np.ndarray, k: int = NEIGHBOURS,
                   exclude: int | None = None) -> list[int]:
    scores = bundle.embeddings @ vector
    if exclude is not None:
        scores[exclude] = -np.inf
    return [int(i) for i in np.argsort(-scores)[:k]]


def embed_image(bundle: Bundle, image: np.ndarray) -> np.ndarray:
    batch = as_batch(bundle, image)
    with torch.no_grad():
        raw = bundle.model.forward_features(batch)
        pooled = bundle.model.forward_head(raw, pre_logits=True).float().cpu().numpy()
    return features.l2_normalize(pooled.astype(np.float64))[0]


def label_text(bundle: Bundle, label: str, language: str) -> str:
    return bundle.glossary.display(label, language)


def family_text(bundle: Bundle, family: str, language: str) -> str:
    return bundle.glossary.family_display(family, language)


def render_predictions(bundle: Bundle, grouped: dict[str, list[tuple[str, float]]],
                       locale: Locale) -> str:
    lines = [f"### {locale.text('predictions.heading')}", ""]
    for family in bundle.schema.families:
        lines.append(f"**{family_text(bundle, family.name, locale.language)}**")
        picks = grouped[family.name]
        if not picks:
            lines.append(f"- _{locale.text('predictions.none')}_")
        for label, probability in picks:
            shown = label_text(bundle, label, locale.language)
            lines.append(f"- {shown} — {probability:.2f} {locale.text('predictions.confidence')}")
        lines.append("")
    return "\n".join(lines)


def render_audit(bundle: Bundle, report: dict[str, dict[str, object]], locale: Locale) -> str:
    lines = [f"### {locale.text('audit.heading')}", ""]
    interesting = 0
    for family in bundle.schema.families:
        entry = report[family.name]
        status = str(entry["status"])
        stored = entry["stored"]
        predicted = [label for label, _ in entry["predicted"]]  # type: ignore[misc]
        if stored is None:
            stored_text = locale.text("audit.unobserved")
        else:
            joined = ", ".join(label_text(bundle, s, locale.language) for s in stored)
            stored_text = joined or locale.text("audit.unobserved")
        predicted_text = ", ".join(label_text(bundle, p, locale.language) for p in predicted)
        heading = f"**{family_text(bundle, family.name, locale.language)}**"
        if status == STATUS_CONFLICT:
            interesting += 1
            detail = locale.text("audit.conflict_detail", stored=stored_text, predicted=predicted_text)
            lines.append(f"{heading} — {locale.text('audit.conflict')}")
            lines.append(f"- {detail}")
        elif status == STATUS_FILL:
            interesting += 1
            lines.append(f"{heading} — {locale.text('audit.fill')}")
            lines.append(f"- {locale.text('audit.fill_detail', predicted=predicted_text)}")
        else:
            lines.append(f"{heading} — {locale.text('audit.agrees')}")
            lines.append(f"- {locale.text('audit.agrees_detail', stored=stored_text)}")
        lines.append("")
    if interesting == 0:
        lines.append(f"_{locale.text('audit.none')}_")
    return "\n".join(lines)


def gallery_items(bundle: Bundle, rows: list[int], locale: Locale) -> list[tuple[np.ndarray, str]]:
    items = []
    for row in rows:
        record = bundle.frame.iloc[row]
        caption = locale.text("neighbours.caption", brand=record["brand"], product=record["product_id"])
        items.append((np.asarray(bundle.images[row]), caption))
    return items


def catalog_choices(bundle: Bundle, limit: int = CATALOG_CHOICES) -> list[tuple[str, int]]:
    frame = bundle.frame.head(limit)
    return [
        (f"{record['brand']} — {record['product_id']}.{record['color_id']}", int(index))
        for index, record in frame.iterrows()
    ]


def load_bundle(run_dir: Path = WINNER_RUN, processed_dir: Path = reporting.PROCESSED_DIR,
                device: torch.device | None = None) -> Bundle:
    glossary = Glossary.load()
    artifacts = data_setup.load_artifacts(processed_dir)
    schema = artifacts.schema
    assert_covers(glossary, schema.columns)
    assert_covers_families(glossary, FAMILIES)
    assert_turkish_complete(glossary)

    payload = evaluate.load_checkpoint(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = evaluate.build_from_checkpoint(payload, len(schema.columns), device)
    config = dict(payload["config"])

    embeddings_path = run_dir / EMBEDDINGS_NAME
    if not embeddings_path.exists():
        raise AppError(
            f"{embeddings_path} is missing. Run retrieval.py for this run so the demo has an index."
        )
    embeddings = np.load(embeddings_path).astype(np.float32)

    thresholds_payload = json.loads(
        (run_dir / reporting.THRESHOLDS_NAME).read_text(encoding="utf-8")
    )
    thresholds = np.array([thresholds_payload[name] for name in schema.columns], dtype=np.float64)
    calibrators = metrics.Calibrators.from_dict(
        json.loads((run_dir / reporting.CALIBRATION_NAME).read_text(encoding="utf-8"))
    )

    transform = data_setup.build_transforms(
        image_size=int(config.get("image_size", 224)),
        mean=tuple(config.get("mean", data_setup.IMAGENET_MEAN)),
        std=tuple(config.get("std", data_setup.IMAGENET_STD)),
        train=False,
    )
    return Bundle(
        model=model,
        schema=schema,
        glossary=glossary,
        calibrators=calibrators,
        thresholds=thresholds,
        embeddings=embeddings,
        frame=pd.read_csv(processed_dir / "catalog.csv"),
        labels=artifacts.labels,
        family_observed=artifacts.family_observed,
        images=np.load(processed_dir / data_setup.IMAGES_NAME, mmap_mode="r"),
        device=device,
        transform=transform,
        run_name=run_dir.name,
    )


def build_interface(bundle: Bundle, locales: dict[str, Locale]):  # noqa: ANN201
    import gradio as gr

    i18n.assert_languages_agree(locales)
    i18n.assert_no_blank_strings(locales)
    choices = catalog_choices(bundle)

    def on_upload(image: np.ndarray | None, language: str):  # noqa: ANN202
        locale = locales[language]
        if image is None:
            return locale.text("upload.no_image"), []
        probabilities = predict_row(bundle, image)
        vector = embed_image(bundle, image)
        rows = neighbours_for(bundle, vector.astype(np.float32))
        return render_predictions(bundle, family_predictions(bundle, probabilities), locale), \
            gallery_items(bundle, rows, locale)

    def on_catalog(row: int | None, language: str):  # noqa: ANN202
        locale = locales[language]
        if row is None:
            return locale.text("catalog.no_selection"), "", []
        image = np.asarray(bundle.images[row])
        probabilities = predict_row(bundle, image)
        report = audit_row(bundle, int(row), probabilities)
        rows = neighbours_for(bundle, bundle.embeddings[row], exclude=int(row))
        return (
            render_predictions(bundle, family_predictions(bundle, probabilities), locale),
            render_audit(bundle, report, locale),
            gallery_items(bundle, rows, locale),
        )

    def header_text(locale: Locale) -> str:
        return f"# {locale.text('app.title')}\n\n{locale.text('app.subtitle')}"

    def upload_help_text(locale: Locale) -> str:
        return f"### {locale.text('upload.heading')}\n\n{locale.text('upload.help')}"

    def catalog_help_text(locale: Locale) -> str:
        return f"### {locale.text('catalog.heading')}\n\n{locale.text('catalog.help')}"

    def footer_text(locale: Locale) -> str:
        return "\n\n".join([
            locale.text("neighbours.help"),
            locale.text("footer.retrieval"),
            locale.text("footer.seed"),
            locale.text("footer.license"),
        ])

    default = locales[DEFAULT_LANGUAGE]
    with gr.Blocks(title=default.text("app.title")) as demo:
        language = gr.Radio(
            choices=[(LANGUAGE_NAMES[code], code) for code in LANGUAGES],
            value=DEFAULT_LANGUAGE,
            label=default.text("app.language"),
        )
        header = gr.Markdown(header_text(default))
        model_note = gr.Markdown(default.text("app.model_note", run=bundle.run_name))

        with gr.Tab(default.text("tab.upload")) as upload_tab:
            upload_help = gr.Markdown(upload_help_text(default))
            upload_image = gr.Image(type="numpy", label=default.text("upload.input"))
            upload_button = gr.Button(default.text("upload.button"))
            upload_predictions = gr.Markdown()
            upload_gallery = gr.Gallery(label=default.text("neighbours.heading"), columns=5)
            upload_button.click(
                on_upload, inputs=[upload_image, language],
                outputs=[upload_predictions, upload_gallery],
            )

        with gr.Tab(default.text("tab.catalog")) as catalog_tab:
            catalog_help = gr.Markdown(catalog_help_text(default))
            picker = gr.Dropdown(choices=choices, label=default.text("catalog.picker"), value=None)
            catalog_button = gr.Button(default.text("catalog.button"))
            catalog_predictions = gr.Markdown()
            catalog_audit = gr.Markdown()
            catalog_gallery = gr.Gallery(label=default.text("neighbours.heading"), columns=5)
            catalog_button.click(
                on_catalog, inputs=[picker, language],
                outputs=[catalog_predictions, catalog_audit, catalog_gallery],
            )

        footer = gr.Markdown(footer_text(default))

        chrome = [
            language, header, model_note, upload_tab, upload_help, upload_image, upload_button,
            upload_gallery, catalog_tab, catalog_help, picker, catalog_button, catalog_gallery, footer,
        ]

        def relabel(selected: str):  # noqa: ANN202
            locale = locales[selected]
            return [
                gr.update(label=locale.text("app.language")),
                gr.update(value=header_text(locale)),
                gr.update(value=locale.text("app.model_note", run=bundle.run_name)),
                gr.update(label=locale.text("tab.upload")),
                gr.update(value=upload_help_text(locale)),
                gr.update(label=locale.text("upload.input")),
                gr.update(value=locale.text("upload.button")),
                gr.update(label=locale.text("neighbours.heading")),
                gr.update(label=locale.text("tab.catalog")),
                gr.update(value=catalog_help_text(locale)),
                gr.update(label=locale.text("catalog.picker")),
                gr.update(value=locale.text("catalog.button")),
                gr.update(label=locale.text("neighbours.heading")),
                gr.update(value=footer_text(locale)),
            ]

        language.change(relabel, inputs=[language], outputs=chrome)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gradio demo")
    parser.add_argument("--run", type=Path, default=WINNER_RUN)
    parser.add_argument("--processed", type=Path, default=reporting.PROCESSED_DIR)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    bundle = load_bundle(args.run, args.processed)
    demo = build_interface(bundle, i18n.load_all())
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
