"""Command-line interface.

    defect-inspector fit samples/metal/train/good --out models/metal.npz
    defect-inspector inspect samples/metal/test/scratch/000.png --model models/metal.npz --heatmap out/heat.png
    defect-inspector batch samples/metal/test --model models/metal.npz --out results.csv --heatmaps out/
    defect-inspector serve
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import Settings


def _vision(settings: Settings, enabled: bool):
    if not enabled:
        return None
    from .describe import get_vision_model

    return get_vision_model(settings)


def _load_model(path: str):
    from .patchcore import PatchCoreModel

    if not Path(path).exists():
        raise FileNotFoundError(f"model file not found: {path} (create it with 'defect-inspector fit')")
    return PatchCoreModel.load(path)


def cmd_fit(args: argparse.Namespace) -> int:
    from .pipeline import fit

    settings = Settings.from_env()
    if args.backbone:
        settings.backbone = args.backbone
    t0 = time.time()
    model = fit(args.folder, settings)
    path = model.save(args.out)
    print(f"fitted on {model.n_train} good images with {model.backbone} in {time.time() - t0:.1f}s; "
          f"memory bank {model.bank.shape[0]} x {model.bank.shape[1]}; threshold {model.threshold:.3f}",
          file=sys.stderr)
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _write_outputs(outcome, heatmap_path: Path | None) -> None:
    if heatmap_path:
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        outcome.overlay.save(heatmap_path)
        outcome.result.heatmap_png = str(heatmap_path)


def cmd_inspect(args: argparse.Namespace) -> int:
    from .pipeline import inspect

    settings = Settings.from_env()
    model = _load_model(args.model)
    outcome = inspect(args.image, model, args.sensitivity, _vision(settings, args.describe))
    _write_outputs(outcome, Path(args.heatmap) if args.heatmap else None)
    text = json.dumps(outcome.result.model_dump(), indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    print(text)
    r = outcome.result
    print(f"{'DEFECT' if r.defect else 'ok'}: score {r.score:.2f} x threshold, {len(r.regions)} region(s)",
          file=sys.stderr)
    return 0


CSV_COLS = ["file", "defect", "score", "raw_score", "threshold", "regions", "top_bbox", "top_peak", "top_label",
            "top_description", "heatmap_png"]


def cmd_batch(args: argparse.Namespace) -> int:
    from .pipeline import inspect, list_images

    settings = Settings.from_env()
    model = _load_model(args.model)
    vision = _vision(settings, args.describe)
    files = list_images(args.folder, args.glob) if not args.recursive else sorted(
        p for p in Path(args.folder).rglob(args.glob) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not files:
        print(f"no images in {args.folder}", file=sys.stderr)
        return 1
    heat_dir = Path(args.heatmaps) if args.heatmaps else None
    results, rows = [], []
    for f in files:
        rel = f.relative_to(args.folder)
        outcome = inspect(f, model, args.sensitivity, vision, name=rel.as_posix())
        if heat_dir:
            _write_outputs(outcome, heat_dir / rel.with_suffix(".heatmap.png"))
        r = outcome.result
        results.append(r.model_dump())
        top = r.regions[0] if r.regions else None
        rows.append({"file": r.file, "defect": int(r.defect), "score": r.score, "raw_score": r.raw_score,
                     "threshold": r.threshold, "regions": len(r.regions),
                     "top_bbox": " ".join(map(str, top.bbox)) if top else "", "top_peak": top.peak if top else "",
                     "top_label": (top.label or "") if top else "", "top_description": (top.description or "") if top else "",
                     "heatmap_png": r.heatmap_png or ""})
        print(f"{rel}: {'DEFECT' if r.defect else 'ok'} {r.score:.2f}", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    flagged = sum(r["defect"] for r in rows)
    print(f"wrote {out} ({flagged} of {len(rows)} images flagged)", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .webapp import launch

    launch(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="defect-inspector", description="Find surface defects from a few good "
                                "samples, with a heatmap of where they are.")
    p.add_argument("--version", action="version", version=f"defect-inspector {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="build a model from a folder of defect-free images")
    f.add_argument("folder")
    f.add_argument("--out", default="models/model.npz", help="model file to write (default: models/model.npz)")
    f.add_argument("--backbone", help="override MS_BACKBONE, e.g. facebook/dinov2-small")
    f.set_defaults(func=cmd_fit)

    def common(sp):
        sp.add_argument("--model", required=True, help="model file from 'fit'")
        sp.add_argument("--sensitivity", type=float, default=1.0,
                        help="above 1 flags more (threshold divided by this), below 1 flags less (default 1.0)")
        sp.add_argument("--describe", action="store_true",
                        help="label and describe flagged regions with the vision model (MS_BACKEND)")

    i = sub.add_parser("inspect", help="score one image; print JSON, optionally save the heatmap PNG")
    i.add_argument("image")
    common(i)
    i.add_argument("--heatmap", help="write the heatmap overlay PNG here")
    i.add_argument("--json", help="also write the JSON result here")
    i.set_defaults(func=cmd_inspect)

    b = sub.add_parser("batch", help="score every image in a folder to one CSV or JSON file")
    b.add_argument("folder")
    common(b)
    b.add_argument("--out", required=True, help="results.csv or results.json")
    b.add_argument("--heatmaps", help="folder for heatmap PNGs (mirrors the input layout)")
    b.add_argument("--glob", default="*", help="which files to read (default: *)")
    b.add_argument("--recursive", action="store_true", help="include sub-folders")
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("serve", help="start the local web demo (Gradio)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
