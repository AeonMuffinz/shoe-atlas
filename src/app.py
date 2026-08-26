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
NEIGHBOURS: int = 5
CATALOG_CHOICES: int = 400
EXAMPLES_DIR: Path = Path("examples")
TOP_PICKS: int = 2
BAR_STRONG: str = "#f97316"
BAR_SOFT: str = "#fdba74"
MODE_UPLOAD: str = "upload"
MODE_CATALOG: str = "catalog"

CSS: str = """
footer { display: none !important; }
.app-head h1 { font-size: 1.5rem !important; margin: 0 0 0.15rem !important; }
.app-head p { font-size: 0.85rem !important; margin: 0.1rem 0 !important; opacity: 0.8; }
.panel-help h3 { font-size: 0.95rem !important; margin: 0 0 0.55rem !important; }
.panel-help p { font-size: 0.82rem !important; margin: 0 !important; opacity: 0.75; }
.mode-switch label { font-size: 0.98rem !important; padding: 0.5rem 1rem !important; }
.mode-switch .wrap { gap: 0.75rem !important; }
.app-status { padding: 0.65rem 0.9rem; border-radius: 8px; font-size: 0.92rem;
              background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.35); }
.pred-note { font-size: 0.78rem; opacity: 0.55; margin-top: 0.6rem; }
.shot .image-container, .shot .upload-container, .shot .image-frame { height: 100% !important; }
.shot .image-frame img { height: 100% !important; width: 100% !important;
                         object-fit: contain !important; }
.pred-head { display: flex; justify-content: space-between; align-items: baseline;
             font-size: 1.02rem; font-weight: 700; margin-bottom: 0.5rem; }
.pred-scale { font-size: 0.78rem; font-weight: 500; opacity: 0.6; }
.pred-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
             gap: 0.45rem 0.7rem; }
.pred-family { border: 1px solid rgba(148,163,184,0.25); border-radius: 10px;
               padding: 0.4rem 0.65rem 0.3rem; }
.pred-family-name { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
                    text-transform: uppercase; opacity: 0.55; margin-bottom: 0.3rem; }
.pred-row { margin-bottom: 0.3rem; }
.pred-line { display: flex; justify-content: space-between; gap: 0.5rem;
             font-size: 0.88rem; margin-bottom: 0.18rem; }
.pred-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pred-value { font-variant-numeric: tabular-nums; opacity: 0.7; flex: none; }
.pred-track { height: 7px; border-radius: 999px; background: rgba(148,163,184,0.22); overflow: hidden; }
.pred-fill { height: 100%; border-radius: 999px; }
.pred-empty { font-size: 0.82rem; opacity: 0.55; font-style: italic; }
.strip { border: 1px solid rgba(249,115,22,0.35) !important; border-radius: 12px !important; }
.strip .grid-wrap { overflow: visible !important; max-height: none !important;
                    min-height: 0 !important; height: auto !important; }
.strip .gallery-container { height: auto !important; }
.strip .gallery-item { height: 190px !important; }
.strip .gallery-item img { height: 100% !important; width: 100% !important;
                           object-fit: contain !important; }
.samples { border: 1px dashed rgba(148,163,184,0.45) !important; border-radius: 12px !important;
           padding: 0.35rem 0.5rem !important; }
.samples .gallery-item, .samples button { height: 100px !important; }
.samples img { height: 100% !important; object-fit: contain !important; }
.progress-text, .progress-level-inner { font-size: 1rem !important; font-weight: 600 !important;
                                        letter-spacing: 0.01em !important; }
.progress-bar-wrap { height: 12px !important; border-radius: 999px !important;
                     background: rgba(148,163,184,0.25) !important; overflow: hidden !important; }
.progress-bar { height: 100% !important; border-radius: 999px !important;
                background: linear-gradient(90deg, #fb923c, #f97316, #ea580c, #f97316) !important;
                background-size: 300% 100% !important;
                animation: bar-shimmer 1.6s linear infinite !important;
                box-shadow: 0 0 10px rgba(249,115,22,0.55) !important; }
@keyframes bar-shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }
@media (max-width: 620px) { .pred-grid { grid-template-columns: minmax(0, 1fr); } }
"""

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


def top_picks(bundle: Bundle, probabilities: np.ndarray, family_name: str,
              limit: int = TOP_PICKS) -> list[tuple[str, float]]:
    family = bundle.schema.family(family_name)
    block = probabilities[family.start : family.end]
    if family.kind == "softmax":
        order = np.argsort(-block)[:limit]
        return [(family.labels[int(i)], float(block[int(i)])) for i in order]
    picks = [
        (label, float(block[i]))
        for i, label in enumerate(family.labels)
        if block[i] >= bundle.thresholds[family.start + i]
    ]
    return sorted(picks, key=lambda item: -item[1])[:limit]


def escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def bar_row(label: str, probability: float, leading: bool) -> str:
    width = max(1.0, min(100.0, probability * 100.0))
    colour = BAR_STRONG if leading else BAR_SOFT
    weight = "600" if leading else "400"
    return (
        f'<div class="pred-row">'
        f'<div class="pred-line"><span class="pred-label" style="font-weight:{weight}">{escape(label)}</span>'
        f'<span class="pred-value">{probability:.2f}</span></div>'
        f'<div class="pred-track">'
        f'<div class="pred-fill" style="width:{width:.1f}%;background:{colour}"></div>'
        f"</div></div>"
    )


def render_predictions(bundle: Bundle, probabilities: np.ndarray, locale: Locale) -> str:
    cards = []
    for family in bundle.schema.families:
        picks = top_picks(bundle, probabilities, family.name)
        rows = "".join(
            bar_row(label_text(bundle, label, locale.language), value, index == 0)
            for index, (label, value) in enumerate(picks)
        )
        if not picks:
            rows = f'<div class="pred-empty">{escape(locale.text("predictions.none"))}</div>'
        cards.append(
            f'<div class="pred-family">'
            f'<div class="pred-family-name">{escape(family_text(bundle, family.name, locale.language))}</div>'
            f"{rows}</div>"
        )
    head = (
        f'<div class="pred-head"><span>{escape(locale.text("predictions.heading"))}</span>'
        f'<span class="pred-scale">{escape(locale.text("predictions.confidence"))}</span></div>'
    )
    return f'{head}<div class="pred-grid">{"".join(cards)}</div>'


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
            lines.append(f"- {heading} — {locale.text('audit.conflict')} · {detail}")
        elif status == STATUS_FILL:
            interesting += 1
            lines.append(f"- {heading} · {locale.text('audit.fill_detail', predicted=predicted_text)}")
        else:
            lines.append(
                f"- {heading} — {locale.text('audit.agrees')} · "
                f"{locale.text('audit.agrees_detail', stored=stored_text)}"
            )
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


def example_images(root: Path = EXAMPLES_DIR) -> list[str]:
    if not root.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [str(p) for p in sorted(root.iterdir()) if p.suffix.lower() in suffixes]


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

    def mode_choices(locale: Locale) -> list[tuple[str, str]]:
        return [
            (locale.text("tab.upload"), MODE_UPLOAD),
            (locale.text("tab.catalog"), MODE_CATALOG),
        ]

    def header_text(locale: Locale) -> str:
        link = f'<a href="{locale.text("app.model_url")}" target="_blank" rel="noopener">' \
               f'{escape(locale.text("app.model_link"))}</a>'
        return (
            f"# {locale.text('app.title')}\n\n{locale.text('app.subtitle')}\n\n"
            f"<sub>{escape(locale.text('app.model_note'))} {link}</sub>"
        )

    def upload_help_text(locale: Locale) -> str:
        return f"### {locale.text('upload.heading')}\n\n{locale.text('upload.help')}"

    def catalog_help_text(locale: Locale) -> str:
        return f"### {locale.text('catalog.heading')}\n\n{locale.text('catalog.help')}"

    def footer_text(locale: Locale) -> str:
        return "\n".join(
            f"- {locale.text(key)}"
            for key in ("neighbours.help", "footer.colour", "notes.resolution", "footer.license")
        )

    def notice(locale: Locale, key: str) -> str:
        return f'<div class="app-status">{escape(locale.text(key))}</div>' 

    def on_upload(  # noqa: ANN202
        image: np.ndarray | None, language: str, progress=gr.Progress()  # noqa: B008, ANN001
    ):
        locale = locales[language]
        if image is None:
            return notice(locale, "upload.no_image"), "", gr.update(visible=False)
        progress(0.05, desc=locale.text("status.working"))
        probabilities = predict_row(bundle, image)
        progress(0.5, desc=locale.text("status.working"))
        vector = embed_image(bundle, image)
        progress(0.8, desc=locale.text("status.working"))
        rows = neighbours_for(bundle, vector.astype(np.float32))
        progress(1.0, desc=locale.text("status.working"))
        return (
            "",
            render_predictions(bundle, probabilities, locale),
            gr.update(value=gallery_items(bundle, rows, locale), visible=True),
        )

    def on_catalog(  # noqa: ANN202
        row: int | None, language: str, progress=gr.Progress()  # noqa: B008, ANN001
    ):
        locale = locales[language]
        if row is None:
            return (
                notice(locale, "catalog.no_selection"), "", "",
                gr.update(visible=False), gr.update(visible=False),
            )
        progress(0.05, desc=locale.text("status.working"))
        image = np.asarray(bundle.images[row])
        probabilities = predict_row(bundle, image)
        progress(0.6, desc=locale.text("status.working"))
        report = audit_row(bundle, int(row), probabilities)
        progress(0.85, desc=locale.text("status.working"))
        rows = neighbours_for(bundle, bundle.embeddings[row], exclude=int(row))
        progress(1.0, desc=locale.text("status.working"))
        return (
            "",
            render_predictions(bundle, probabilities, locale),
            render_audit(bundle, report, locale),
            gr.update(value=gallery_items(bundle, rows, locale), visible=True),
            gr.update(value=image, visible=True),
        )

    default = locales[DEFAULT_LANGUAGE]
    with gr.Blocks(title=default.text("app.title")) as demo:
        with gr.Row():
            header = gr.Markdown(header_text(default), elem_classes=["app-head"])
            language = gr.Radio(
                choices=[(LANGUAGE_NAMES[code], code) for code in LANGUAGES],
                value=DEFAULT_LANGUAGE, label=default.text("app.language"), scale=0,
            )
        mode = gr.Radio(
            choices=mode_choices(default), value=MODE_UPLOAD,
            show_label=False, elem_classes=["mode-switch"],
        )

        with gr.Group(visible=True) as upload_panel:
            upload_help = gr.Markdown(upload_help_text(default), elem_classes=["panel-help"])
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    upload_image = gr.Image(
                        type="numpy", label=default.text("upload.input"), height=300,
                        placeholder=default.text("upload.placeholder"),
                        sources=["upload", "clipboard"],
                    )
                    upload_button = gr.Button(default.text("upload.button"), variant="primary")
                with gr.Column(scale=5):
                    upload_status = gr.HTML()
                    upload_predictions = gr.HTML()
            upload_gallery = gr.Gallery(
                label=default.text("neighbours.heading"), columns=5, rows=1,
                object_fit="contain", visible=False, elem_classes=["strip"],
            )
            samples = example_images()
            examples = None
            if samples:
                with gr.Group(elem_classes=["samples"]):
                    examples = gr.Examples(
                        examples=samples, inputs=upload_image,
                        label=default.text("examples.heading"),
                    )

        with gr.Group(visible=False) as catalog_panel:
            catalog_help = gr.Markdown(catalog_help_text(default), elem_classes=["panel-help"])
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    picker = gr.Dropdown(
                        choices=choices, label=default.text("catalog.picker"), value=None,
                    )
                    catalog_button = gr.Button(default.text("catalog.button"), variant="primary")
                    catalog_preview = gr.Image(
                        label=default.text("catalog.preview"), height=320, visible=False,
                        elem_classes=["shot"],
                    )
                with gr.Column(scale=5):
                    catalog_status = gr.HTML()
                    catalog_predictions = gr.HTML()
            catalog_gallery = gr.Gallery(
                label=default.text("neighbours.heading"), columns=5, rows=1,
                object_fit="contain", visible=False, elem_classes=["strip"],
            )
            catalog_audit = gr.Markdown()

        footer = gr.Markdown(footer_text(default))

        upload_button.click(
            lambda: ("", "", gr.update(visible=False)),
            outputs=[upload_status, upload_predictions, upload_gallery],
            show_progress="hidden",
        ).then(
            on_upload, inputs=[upload_image, language],
            outputs=[upload_status, upload_predictions, upload_gallery],
            show_progress="full", show_progress_on=[upload_predictions],
        )
        catalog_button.click(
            lambda: ("", "", "", gr.update(visible=False), gr.update(visible=False)),
            outputs=[catalog_status, catalog_predictions, catalog_audit,
                     catalog_gallery, catalog_preview],
            show_progress="hidden",
        ).then(
            on_catalog, inputs=[picker, language],
            outputs=[catalog_status, catalog_predictions, catalog_audit,
                     catalog_gallery, catalog_preview],
            show_progress="full", show_progress_on=[catalog_predictions],
        )

        def switch_mode(selected: str):  # noqa: ANN202
            return (
                gr.update(visible=selected == MODE_UPLOAD),
                gr.update(visible=selected == MODE_CATALOG),
            )

        mode.change(switch_mode, inputs=[mode], outputs=[upload_panel, catalog_panel])

        chrome = [
            language, mode, header, upload_help, upload_image, upload_button, upload_gallery,
            catalog_help, picker, catalog_button, catalog_preview, catalog_gallery, footer,
        ]
        if examples is not None:
            chrome.append(examples.dataset)

        def relabel(selected: str, current_mode: str):  # noqa: ANN202
            locale = locales[selected]
            return [
                gr.update(label=locale.text("app.language")),
                gr.update(choices=mode_choices(locale), value=current_mode),
                gr.update(value=header_text(locale)),
                gr.update(value=upload_help_text(locale)),
                gr.update(label=locale.text("upload.input"),
                          placeholder=locale.text("upload.placeholder")),
                gr.update(value=locale.text("upload.button")),
                gr.update(label=locale.text("neighbours.heading")),
                gr.update(value=catalog_help_text(locale)),
                gr.update(label=locale.text("catalog.picker")),
                gr.update(value=locale.text("catalog.button")),
                gr.update(label=locale.text("catalog.preview")),
                gr.update(label=locale.text("neighbours.heading")),
                gr.update(value=footer_text(locale)),
                *([gr.update(label=locale.text("examples.heading"))] if examples is not None else []),
            ]

        language.change(relabel, inputs=[language, mode], outputs=chrome)
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
    demo.launch(share=args.share, server_port=args.port, css=CSS, footer_links=[])


if __name__ == "__main__":
    main()
