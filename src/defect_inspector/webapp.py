"""Local web demo (Gradio): pick good samples, inspect an image, see the score, heatmap and regions."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import __version__
from .config import Settings
from .features import get_extractor
from .patchcore import PatchCoreModel
from .pipeline import JSONCache, fit_images, inspect, list_images, open_image, sample_model
from .synthetic import SURFACES

PANEL = 400


def samples_dir() -> Path:
    for d in (Path.cwd() / "samples", Path(__file__).resolve().parents[2] / "samples"):
        if d.is_dir():
            return d
    return Path.cwd() / "samples"


def test_images(surface: str) -> list[str]:
    root = samples_dir() / surface / "test"
    if not root.is_dir():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.png"))


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def side_by_side(original: Image.Image, overlay: Image.Image) -> Image.Image:
    """Original and heatmap overlay next to each other, with small captions."""
    a = original.convert("RGB").resize((PANEL, PANEL), Image.NEAREST)
    b = overlay.convert("RGB").resize((PANEL, PANEL), Image.NEAREST)
    gap, top = 14, 24
    sheet = Image.new("RGB", (PANEL * 2 + gap, PANEL + top), (250, 247, 240))
    sheet.paste(a, (0, top))
    sheet.paste(b, (PANEL + gap, top))
    d = ImageDraw.Draw(sheet)
    f = _font(13)
    d.text((2, 4), "Image", font=f, fill=(107, 102, 94))
    d.text((PANEL + gap + 2, 4), "Heatmap and flagged regions", font=f, fill=(107, 102, 94))
    return sheet


def good_strip(images: list[Image.Image], side: int = 64) -> Image.Image:
    n = len(images)
    strip = Image.new("RGB", (n * side + (n - 1) * 6, side), (250, 247, 240))
    for i, im in enumerate(images):
        strip.paste(im.convert("RGB").resize((side, side)), (i * (side + 6), 0))
    return strip


def region_rows(result) -> list[list]:
    return [[r.id, f"{r.peak:.2f}", " ".join(map(str, r.bbox)), r.area_px, r.label or "", r.description or ""]
            for r in result.regions]


def verdict_markdown(result, model: PatchCoreModel, note: str = "") -> str:
    head = ("**Defect found**" if result.defect else "**No defect found**")
    lines = [f"{head} · score **{result.score:.2f}** × threshold "
             f"(1.00 is the highest score among the good samples, sensitivity {result.sensitivity:.2f})",
             f"{len(result.regions)} flagged region(s)."]
    if note:
        lines.append(note)
    return "  \n".join(lines)


CSS = """
:root, .gradio-container { --ms-ink:#1c1a17; --ms-rust:#8a3a24; --ms-paper:#f3efe6; --ms-cream:#faf7f0; }
.gradio-container { background: var(--ms-cream) !important; font-family: Georgia, 'Times New Roman', serif !important; max-width: 1240px !important; }
.ms-header { border-bottom: 1px solid var(--ms-ink); padding: 10px 4px 12px; margin-bottom: 4px; }
.ms-brand { letter-spacing: .22em; font-size: 12px; color: var(--ms-rust); font-weight: 700; }
.ms-header h1 { font-family: Georgia, serif; font-weight: 400; font-size: 30px; margin: 2px 0 2px; color: var(--ms-ink); }
.ms-header p { margin: 0; color: #4a463f; font-style: italic; }
.ms-meta { margin-top: 8px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #6b665e; }
.ms-muted { color: #6b665e; font-size: 12px; }
.ms-card { background: #fff !important; border: 1px solid #ddd5c6 !important; border-radius: 10px !important; }
.ms-section-title { font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--ms-rust); margin: 2px 0; }
.ms-left { max-height: calc(100vh - 150px); min-height: 560px; overflow-y: auto !important; padding-right: 8px; scroll-behavior: smooth; flex-wrap: nowrap !important; }
.ms-left > * { flex-shrink: 0 !important; }
.ms-table * { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif !important; font-size: 12.5px !important; }
.ms-status { font-size: 14px; }
.ms-secondary, button.ms-secondary { background: #fff !important; color: var(--ms-rust) !important; border: 1px solid var(--ms-rust) !important; }
button.primary, .ms-primary button { background: var(--ms-rust) !important; border-color: var(--ms-rust) !important; color: #fff !important; }
footer { display: none !important; }
"""


def build_app(settings: Settings | None = None):
    import gradio as gr

    settings = settings or Settings.from_env()
    describe_cache = JSONCache(Path(settings.cache_dir) / "describe")
    models: dict[str, PatchCoreModel] = {}
    scores: dict[tuple, object] = {}
    out_dir = Path(tempfile.mkdtemp(prefix="defect-inspector-"))

    def get_model(key: str) -> PatchCoreModel:
        if key not in models:
            models[key] = sample_model(key, settings, samples_dir())
        return models[key]

    def pick_surface(surface):
        model = get_model(surface)
        goods = [open_image(p) for p in list_images(samples_dir() / surface / "train" / "good")]
        choices = test_images(surface)
        msg = (f"Model ready: fitted on **{model.n_train}** good `{surface}` samples "
               f"({model.bank.shape[0]} patch vectors kept). Pick an image to inspect.")
        return (surface, good_strip(goods), gr.update(choices=choices, value=None), msg, None, [], None, None)

    def fit_upload(files):
        paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in (files or [])]
        images = [open_image(p) for p in paths if p]
        if len(images) < 2:
            return gr.update(), gr.update(), "Upload at least 2 defect-free images to fit your own model."
        digest = hashlib.sha256(b"".join(im.tobytes()[:4096] for im in images)).hexdigest()[:10]
        key = f"upload-{digest}"
        models[key] = fit_images(images, settings)
        return (key, good_strip(images[:10]),
                f"Model ready: fitted on **{len(images)}** uploaded good images. Upload an image to inspect.")

    def pick_test(surface_key, name):
        if not name or surface_key not in SURFACES:
            return gr.update()
        return str(samples_dir() / surface_key / "test" / name)

    def pick_upload(file):
        return (file if isinstance(file, str) else getattr(file, "name", None)), gr.update(value=None)

    def run(model_key, path, sensitivity, with_vlm=False):
        if not model_key:
            return None, "Choose a surface or upload good images first.", [], None, None
        if not path:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        model = get_model(model_key) if model_key in SURFACES else models[model_key]
        image = open_image(path)
        skey = (model_key, hashlib.sha256(image.tobytes()).hexdigest())
        if skey not in scores:
            scores[skey] = model.score(image, get_extractor(model.backbone, model.input_size))
        note = ""
        vision = None
        if with_vlm:
            from .describe import get_vision_model

            try:
                vision = get_vision_model(settings)
            except (RuntimeError, ValueError) as exc:
                note = f"⚠️ Could not load the vision model: {exc}"
        outcome = inspect(image, model, sensitivity, vision=vision, name=Path(path).name, score=scores[skey],
                          describe_cache=describe_cache)
        if vision is not None:
            note = (f"Regions described by `{vision.name}` ({settings.backend} backend)."
                    if outcome.result.regions else "Nothing flagged, so nothing to describe.")
        stem = Path(path).stem
        png = out_dir / f"{stem}.heatmap.png"
        outcome.overlay.save(png)
        outcome.result.heatmap_png = png.name
        js = out_dir / f"{stem}.json"
        js.write_text(json.dumps(outcome.result.model_dump(), indent=2) + "\n")
        return (side_by_side(image, outcome.overlay), verdict_markdown(outcome.result, model, note),
                region_rows(outcome.result), str(png), str(js))

    with gr.Blocks(title="defect-inspector · Modelshifts") as demo:
        gr.HTML(
            "<div class='ms-header'><div class='ms-brand'>MODELSHIFTS</div><h1>defect-inspector</h1>"
            "<p>Find surface defects from a few good samples, with a heatmap of where they are.</p>"
            f"<div class='ms-meta'>backbone: {settings.backbone} · describe: {settings.backend} "
            f"({settings.vlm_name or 'not set'}) · v{__version__}</div></div>"
        )
        model_key = gr.State(None)
        current = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=400, elem_classes="ms-left", elem_id="ms-left"):
                gr.HTML("<div class='ms-section-title'>1 · Good samples</div>")
                surface = gr.Radio(list(SURFACES), value=None, label="Synthetic surface")
                with gr.Accordion("…or fit on your own good images", open=False):
                    goods_up = gr.File(file_count="multiple", file_types=["image"], label="Defect-free images (2+)")
                    fit_btn = gr.Button("Fit on uploaded images", elem_classes="ms-secondary")
                strip = gr.Image(type="pil", show_label=False, height=72, interactive=False, container=False)
                gr.HTML("<div class='ms-section-title'>2 · Image to inspect</div>")
                with gr.Row(equal_height=True):
                    test = gr.Dropdown([], value=None, label="Synthetic test image", interactive=True, scale=3)
                    upload = gr.UploadButton("Upload image", file_types=["image"], type="filepath", scale=1,
                                             elem_classes="ms-secondary")
                sens = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Sensitivity (higher flags more)")
                with gr.Row():
                    go = gr.Button("Inspect", variant="primary", elem_classes="ms-primary")
                    desc_btn = gr.Button("Describe regions", elem_classes="ms-secondary")
                status = gr.Markdown("Choose a surface to fit a model on its good samples.", elem_classes="ms-status")
                gr.HTML("<div class='ms-section-title'>3 · Flagged regions</div>")
                table = gr.Dataframe(headers=["#", "Peak", "Box (x0 y0 x1 y1)", "Area", "Label", "Description"],
                                     datatype=["number", "str", "str", "number", "str", "str"], interactive=False,
                                     wrap=True, column_widths=["6%", "11%", "21%", "10%", "13%", "39%"],
                                     elem_classes="ms-table")
                with gr.Row():
                    dl_png = gr.DownloadButton("Heatmap PNG", elem_classes="ms-secondary")
                    dl_json = gr.DownloadButton("Result JSON", elem_classes="ms-secondary")
            with gr.Column(scale=6, min_width=480):
                gr.HTML("<div class='ms-section-title'>Inspection · colour shows how far each area is from the "
                        "good samples, boxes are flagged regions</div>")
                view = gr.Image(type="pil", show_label=False, height=340, interactive=False,
                                elem_classes="ms-card", buttons=["fullscreen", "download"])
        gr.HTML("<p class='ms-muted'>All sample images are procedurally generated. Uploaded files stay on the "
                "machine running this demo.</p>")

        outs = [view, status, table, dl_png, dl_json]
        surface.change(pick_surface, [surface],
                       [model_key, strip, test, status, view, table, dl_png, dl_json])
        fit_btn.click(fit_upload, [goods_up], [model_key, strip, status])
        test.change(pick_test, [model_key, test], [current]).then(run, [model_key, current, sens], outs)
        upload.upload(pick_upload, [upload], [current, test]).then(run, [model_key, current, sens], outs)
        go.click(run, [model_key, current, sens], outs)
        sens.release(run, [model_key, current, sens], outs)
        desc_btn.click(lambda k, p, s: run(k, p, s, True), [model_key, current, sens], outs)
    return demo


def theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.Font("Georgia"), gr.themes.Font("Times New Roman"), gr.themes.Font("serif")],
        font_mono=[gr.themes.Font("ui-monospace"), gr.themes.Font("Menlo"), gr.themes.Font("monospace")],
        radius_size=gr.themes.sizes.radius_md,
    )


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_app()
    demo.queue().launch(server_name=host, server_port=port, share=share, theme=theme(), css=CSS)
