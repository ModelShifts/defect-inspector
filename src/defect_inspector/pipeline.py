"""High-level helpers used by the CLI and the web demo."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .features import get_extractor
from .models import Inspection
from .patchcore import PatchCoreModel, Score
from .regions import crop_region, find_regions, overlay

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str | Path, pattern: str = "*") -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    return sorted(p for p in folder.glob(pattern) if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def fit(folder: str | Path, settings: Settings) -> PatchCoreModel:
    files = list_images(folder)
    if len(files) < 2:
        raise ValueError(f"{folder}: need at least 2 good images to fit, found {len(files)}")
    extractor = get_extractor(settings.backbone, settings.image_size)
    return PatchCoreModel.fit([open_image(f) for f in files], extractor, settings.coreset_ratio)


def fit_images(images: list[Image.Image], settings: Settings) -> PatchCoreModel:
    extractor = get_extractor(settings.backbone, settings.image_size)
    return PatchCoreModel.fit(images, extractor, settings.coreset_ratio)


@dataclass
class Outcome:
    result: Inspection
    heatmap: np.ndarray
    overlay: Image.Image


def inspect(image: Image.Image | str | Path, model: PatchCoreModel, sensitivity: float = 1.0,
            vision=None, max_described: int = 3, name: str | None = None, score: Score | None = None,
            describe_cache=None) -> Outcome:
    """Score one image; optionally describe the top regions with a vision model.

    ``score`` reuses an earlier :meth:`PatchCoreModel.score` result (the web demo does this when only the
    sensitivity changes). ``describe_cache`` is an optional object with ``get(key)`` / ``put(key, value)``.
    """
    if not isinstance(image, Image.Image):
        name = name or Path(image).name
        image = open_image(image)
    s = score or model.score(image, get_extractor(model.backbone, model.input_size))
    regions = find_regions(s.heatmap, model.threshold, sensitivity)
    flagged = s.score * sensitivity >= model.threshold
    if not flagged:
        regions = []
    if vision is not None:
        from .describe import describe

        for r in regions[:max_described]:
            crop = crop_region(image, r)
            key = hashlib.sha256(crop.tobytes() + vision.name.encode()).hexdigest()[:20]
            cached = describe_cache.get(key) if describe_cache else None
            if cached:
                r.label, r.description = cached
            else:
                r.label, r.description = describe(crop, vision)
                if describe_cache:
                    describe_cache.put(key, [r.label, r.description])
    result = Inspection(
        file=name or "image", width=image.width, height=image.height,
        score=round(s.score / model.threshold, 4), raw_score=round(s.score, 4), threshold=round(model.threshold, 4),
        sensitivity=sensitivity, defect=bool(flagged), regions=regions, backbone=model.backbone,
        describer=getattr(vision, "name", None),
    )
    return Outcome(result, s.heatmap, overlay(image, s.heatmap, model.threshold, regions, sensitivity))


def sample_model(surface: str, settings: Settings, samples: Path) -> PatchCoreModel:
    """Fit (once) and cache a model on ``samples/<surface>/train/good``."""
    folder = samples / surface / "train" / "good"
    digest = hashlib.sha256()
    for f in list_images(folder):
        digest.update(f.read_bytes())
    digest.update(f"{settings.backbone}|{settings.image_size}|{settings.coreset_ratio}".encode())
    path = Path(settings.cache_dir) / "models" / f"{surface}-{digest.hexdigest()[:12]}.npz"
    if path.exists():
        return PatchCoreModel.load(path)
    model = fit(folder, settings)
    model.save(path)
    return model


class JSONCache:
    """Small JSON-file cache (used by the web demo for vision-model replies)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, key: str):
        import json

        p = self.root / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, value) -> None:
        import json

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_text(json.dumps(value))
