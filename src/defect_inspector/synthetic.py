"""Procedural textured surfaces with injected defects and pixel masks.

Everything here is generated from a seed: no photos, no third-party data.
Three surface types (``metal``, ``fabric``, ``stone``) and three defect types
(``scratch``, ``dent``, ``spot``). The folder layout follows the common
``train/good``, ``test/<type>``, ``ground_truth/<type>`` convention:

    <root>/<surface>/train/good/000.png
    <root>/<surface>/test/good/000.png
    <root>/<surface>/test/scratch/000.png
    <root>/<surface>/ground_truth/scratch/000_mask.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SURFACES = ("metal", "fabric", "stone")
DEFECTS = ("scratch", "dent", "spot")
SIZE = 256


# ---------------------------------------------------------------- surfaces

def _noise(rng: np.random.Generator, shape, sigma) -> np.ndarray:
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return n / (n.std() + 1e-8)


def _lighting(rng: np.random.Generator, size: int, amount: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size] / size
    gx, gy = rng.uniform(-1, 1, 2)
    return 1.0 + amount * (gx * (x - 0.5) + gy * (y - 0.5))


def metal(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Brushed metal: fine horizontal grain, a few longer streaks, soft lighting."""
    grain = ndimage.gaussian_filter(rng.standard_normal((size, size)), (0.6, 14), mode="wrap")
    grain /= grain.std() + 1e-8
    streak = ndimage.gaussian_filter(rng.standard_normal((size, size)), (1.5, 60), mode="wrap")
    streak /= streak.std() + 1e-8
    g = 0.58 + 0.035 * grain + 0.025 * streak
    g = g * _lighting(rng, size, 0.12)
    tint = np.array([0.97, 0.99, 1.03])
    return np.clip(g[..., None] * tint, 0, 1)


def fabric(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Plain-weave fabric: two thread directions with slight irregularity."""
    y, x = np.mgrid[0:size, 0:size].astype(float)
    period = rng.uniform(7.5, 8.5)
    jitter_x = 1.2 * _noise(rng, (size, size), 10)
    jitter_y = 1.2 * _noise(rng, (size, size), 10)
    px, py = rng.uniform(0, 2 * np.pi, 2)
    warp = 0.5 + 0.5 * np.sin(2 * np.pi * (x + jitter_x) / period + px)
    weft = 0.5 + 0.5 * np.sin(2 * np.pi * (y + jitter_y) / period + py)
    checker = np.sign(np.sin(np.pi * (x + jitter_x) / period + px) * np.sin(np.pi * (y + jitter_y) / period + py))
    weave = np.where(checker > 0, warp, weft)
    fuzz = _noise(rng, (size, size), 0.8)
    g = 0.45 + 0.18 * weave + 0.03 * fuzz
    g = g * _lighting(rng, size, 0.08)
    base = np.array([0.36, 0.45, 0.62]) * rng.uniform(0.95, 1.05)
    return np.clip(g[..., None] * base / 0.55, 0, 1)


def stone(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """Speckled stone tile: multi-scale noise with small dark and light grains."""
    coarse = _noise(rng, (size, size), 18)
    mid = _noise(rng, (size, size), 4)
    fine = _noise(rng, (size, size), 0.7)
    g = 0.66 + 0.04 * coarse + 0.035 * mid + 0.03 * fine
    grains = rng.random((size, size))
    g = g - 0.12 * (grains > 0.992) + 0.07 * (grains < 0.006)
    g = ndimage.gaussian_filter(g, 0.5)
    g = g * _lighting(rng, size, 0.08)
    base = np.array([1.0, 0.95, 0.86])
    return np.clip(g[..., None] * base, 0, 1)


SURFACE_FN = {"metal": metal, "fabric": fabric, "stone": stone}


# ----------------------------------------------------------------- defects

def _centre(rng: np.random.Generator, size: int, margin: int = 40) -> tuple[float, float]:
    margin = min(margin, size // 4)
    return tuple(rng.uniform(margin, size - margin, 2))  # type: ignore[return-value]


def add_scratch(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A thin curved line, lighter or darker than the surface."""
    size = img.shape[0]
    cx, cy = _centre(rng, size)
    length = rng.uniform(50, 110)
    angle = rng.uniform(0, np.pi)
    bend = rng.uniform(-0.5, 0.5)
    t = np.linspace(-0.5, 0.5, 400)
    xs = cx + length * t * np.cos(angle) + bend * length * (t**2 - 0.25) * -np.sin(angle)
    ys = cy + length * t * np.sin(angle) + bend * length * (t**2 - 0.25) * np.cos(angle)
    line = np.zeros((size, size))
    xi, yi = np.clip(np.round(xs).astype(int), 0, size - 1), np.clip(np.round(ys).astype(int), 0, size - 1)
    line[yi, xi] = 1.0
    width = rng.uniform(0.7, 1.3)
    soft = ndimage.gaussian_filter(line, width)
    soft /= soft.max() + 1e-8
    sign = 1 if rng.random() < 0.6 else -1
    strength = rng.uniform(0.16, 0.26)
    out = np.clip(img + sign * strength * soft[..., None], 0, 1)
    mask = soft > 0.2
    return out, mask


def add_dent(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A shallow round dent: a shaded bump lit from one side."""
    size = img.shape[0]
    cx, cy = _centre(rng, size)
    r = rng.uniform(9, 16)
    y, x = np.mgrid[0:size, 0:size].astype(float)
    depth = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2)))
    gy, gx = np.gradient(depth)
    light = rng.uniform(0, 2 * np.pi)
    shade = (gx * np.cos(light) + gy * np.sin(light)) * r
    strength = rng.uniform(0.35, 0.5)
    out = np.clip(img * (1 + strength * shade[..., None]) - 0.04 * depth[..., None], 0, 1)
    mask = depth > 0.25
    return out, mask


def add_spot(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A stain: an irregular soft blob with a brownish tint."""
    size = img.shape[0]
    cx, cy = _centre(rng, size)
    rx, ry = rng.uniform(5, 12, 2)
    y, x = np.mgrid[0:size, 0:size].astype(float)
    rot = rng.uniform(0, np.pi)
    dx, dy = x - cx, y - cy
    u, v = dx * np.cos(rot) + dy * np.sin(rot), -dx * np.sin(rot) + dy * np.cos(rot)
    d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2) + 0.25 * _noise(rng, (size, size), 3)
    alpha = np.clip(1.4 - d, 0, 1)
    alpha = ndimage.gaussian_filter(alpha, 1.0)
    colour = np.array([0.30, 0.22, 0.12]) * rng.uniform(0.7, 1.2)
    opacity = rng.uniform(0.5, 0.7)
    out = img * (1 - opacity * alpha[..., None]) + opacity * alpha[..., None] * colour
    mask = alpha > 0.3
    return np.clip(out, 0, 1), mask


DEFECT_FN = {"scratch": add_scratch, "dent": add_dent, "spot": add_spot}


# ------------------------------------------------------------------ dataset

def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(arr * 255).astype(np.uint8)).save(path, optimize=True)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(path, optimize=True)


def make_dataset(root: str | Path, n_train: int = 8, n_test_good: int = 6, n_test_defect: int = 4,
                 surfaces=SURFACES, seed: int = 7, size: int = SIZE) -> dict:
    """Write the dataset under ``root`` and return a manifest (also saved as manifest.json)."""
    root = Path(root)
    manifest: dict = {"seed": seed, "size": size, "surfaces": {}, "note": "procedurally generated, no real data"}
    for si, surface in enumerate(surfaces):
        rng = np.random.default_rng(seed * 1000 + si)
        fn = SURFACE_FN[surface]
        entry = {"train": [], "test": []}
        for i in range(n_train):
            p = root / surface / "train" / "good" / f"{i:03d}.png"
            _save(fn(rng, size), p)
            entry["train"].append(p.relative_to(root).as_posix())
        for i in range(n_test_good):
            p = root / surface / "test" / "good" / f"{i:03d}.png"
            _save(fn(rng, size), p)
            entry["test"].append({"image": p.relative_to(root).as_posix(), "label": "good", "mask": None})
        for defect in DEFECTS:
            for i in range(n_test_defect):
                img, mask = DEFECT_FN[defect](fn(rng, size), rng)
                p = root / surface / "test" / defect / f"{i:03d}.png"
                m = root / surface / "ground_truth" / defect / f"{i:03d}_mask.png"
                _save(img, p)
                _save_mask(mask, m)
                entry["test"].append({"image": p.relative_to(root).as_posix(), "label": defect,
                                      "mask": m.relative_to(root).as_posix()})
        manifest["surfaces"][surface] = entry
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
