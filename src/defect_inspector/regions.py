"""Turn a heatmap into regions and draw the overlay image."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .models import Region

RUST = (138, 58, 36)
INK = (28, 26, 23)

# colour ramp for the overlay: transparent -> amber -> rust -> deep red-brown
_STOPS = np.array([
    [0.00, 231, 196, 92, 0],
    [0.25, 231, 196, 92, 70],
    [0.60, 214, 110, 46, 120],
    [1.00, 138, 30, 20, 160],
], dtype=np.float32)


def find_regions(heatmap: np.ndarray, threshold: float, sensitivity: float = 1.0,
                 min_area: int = 12, max_regions: int = 10, core: float = 0.6) -> list[Region]:
    """Connected areas above the cut, highest peak first.

    The cut is ``threshold / sensitivity`` (at most 90% of the image's highest value), raised to ``core`` times the image's highest value so a
    region outlines the core of an anomaly rather than its smoothed halo.
    """
    top = float(heatmap.max())
    # smoothing lowers the peak a little, so a flagged image always keeps its strongest area
    cut = max(min(threshold / max(sensitivity, 1e-6), 0.9 * top), core * top)
    labels, n = ndimage.label(heatmap >= cut)
    regions = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        mask = labels[sl] == i
        area = int(mask.sum())
        if area < min_area:
            continue
        peak = float(heatmap[sl][mask].max()) / threshold
        regions.append((peak, area, [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]))
    regions.sort(key=lambda r: -r[0])
    return [Region(id=k, bbox=b, area_px=a, peak=round(p, 3))
            for k, (p, a, b) in enumerate(regions[:max_regions], start=1)]


def colourise(heatmap: np.ndarray, threshold: float, sensitivity: float = 1.0) -> Image.Image:
    """RGBA overlay. Values below 80% of the cut stay clear; colour deepens up to twice the cut."""
    cut = threshold / max(sensitivity, 1e-6)
    t = np.clip((heatmap / cut - 0.8) / 1.2, 0, 1)
    out = np.zeros(heatmap.shape + (4,), dtype=np.float32)
    for ch in range(4):
        out[..., ch] = np.interp(t, _STOPS[:, 0], _STOPS[:, ch + 1])
    return Image.fromarray(out.round().astype(np.uint8), "RGBA")


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(image: Image.Image, heatmap: np.ndarray, threshold: float, regions: list[Region],
            sensitivity: float = 1.0, scale: int | None = None) -> Image.Image:
    """The image with the heatmap blended on top and numbered boxes around regions."""
    base = image.convert("RGBA")
    out = Image.alpha_composite(base, colourise(heatmap, threshold, sensitivity))
    if scale is None:
        scale = max(1, 512 // max(out.size))
    if scale > 1:
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(out)
    font = _font(11 * scale if scale < 3 else 24)
    for r in regions:
        x0, y0, x1, y1 = (v * scale for v in r.bbox)
        d.rectangle((x0 - 2, y0 - 2, x1 + 1, y1 + 1), outline=(255, 255, 255, 255), width=max(1, scale))
        d.rectangle((x0 - 3, y0 - 3, x1 + 2, y1 + 2), outline=RUST + (255,), width=max(1, scale))
        label = str(r.id)
        bw, bh = d.textlength(label, font=font) + 8 * scale // 2 + 6, 16 * scale // 2 + 8
        bx, by = x0 - 3, y0 - 3 - bh
        if by < 0:
            by = y1 + 3
        d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=3, fill=RUST + (255,))
        d.text((bx + 5, by + 2), label, font=font, fill=(255, 255, 255, 255))
    return out.convert("RGB")


def crop_region(image: Image.Image, region: Region, min_side: int = 72, out_side: int = 384) -> Image.Image:
    """A square crop around a region with some context, resized for a vision model."""
    x0, y0, x1, y1 = region.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(min_side, int(1.8 * max(x1 - x0, y1 - y0)))
    side = min(side, max(image.size))
    left = int(np.clip(cx - side / 2, 0, max(0, image.width - side)))
    top = int(np.clip(cy - side / 2, 0, max(0, image.height - side)))
    crop = image.convert("RGB").crop((left, top, left + side, top + side))
    return crop.resize((out_side, out_side), Image.LANCZOS)
