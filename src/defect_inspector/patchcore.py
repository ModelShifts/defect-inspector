"""PatchCore (Roth et al., CVPR 2022), written from the paper.

Fit: extract patch features from a few defect-free images and keep a coreset
of them (greedy k-centre selection) as the memory bank.
Score: each patch of a new image is scored by its distance to the nearest
patch in the memory bank. The image score is the highest patch score, and the
patch scores, upsampled and smoothed, form the heatmap.

The decision threshold is set from the good images alone: every training image
is scored against a memory bank built from the *other* training images
(leave-one-out), and the threshold is the highest of those scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .features import Extractor

FORMAT_VERSION = 1


def nearest_distance(queries: np.ndarray, bank: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Euclidean distance from each query row to its nearest bank row."""
    import torch

    q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32))
    b = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32))
    out = []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], b)
        out.append(d.min(dim=1).values)
    return torch.cat(out).numpy()


def greedy_coreset(features: np.ndarray, n: int, proj_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Indices of ``n`` rows chosen by greedy k-centre on a random projection."""
    import torch

    if n >= len(features):
        return np.arange(len(features))
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(features.astype(np.float32))
    if x.shape[1] > proj_dim:
        proj = torch.from_numpy((rng.standard_normal((x.shape[1], proj_dim)) / np.sqrt(proj_dim)).astype(np.float32))
        x = x @ proj
    selected = [int(rng.integers(len(x)))]
    min_d = torch.cdist(x, x[selected[0]:selected[0] + 1]).squeeze(1)
    for _ in range(n - 1):
        idx = int(torch.argmax(min_d))
        selected.append(idx)
        min_d = torch.minimum(min_d, torch.cdist(x, x[idx:idx + 1]).squeeze(1))
    return np.array(selected)


@dataclass
class Score:
    score: float  # raw image score (max patch distance)
    patch_map: np.ndarray  # (h, w) patch distances
    heatmap: np.ndarray  # (H, W) float32 at image resolution


@dataclass
class PatchCoreModel:
    backbone: str
    input_size: int
    bank: np.ndarray
    threshold: float
    train_scores: list[float] = field(default_factory=list)
    n_train: int = 0
    grid: tuple[int, int] = (0, 0)
    sigma: float = 4.0

    # -------------------------------------------------------------- fit
    @classmethod
    def fit(cls, images: list[Image.Image], extractor: Extractor, coreset_ratio: float = 0.1,
            seed: int = 0) -> "PatchCoreModel":
        if len(images) < 2:
            raise ValueError("fit needs at least 2 good images (leave-one-out sets the threshold)")
        feats = extractor(images)  # N, h, w, C
        n, h, w, c = feats.shape
        per_image = feats.reshape(n, h * w, c)
        per_size = max(1, int(round(h * w * coreset_ratio)))
        # leave-one-out scores of the training images set the threshold
        loo = []
        for i in range(n):
            others = np.concatenate([per_image[j] for j in range(n) if j != i])
            sub = others[greedy_coreset(others, per_size * (n - 1), seed=seed)]
            loo.append(float(nearest_distance(per_image[i], sub).max()))
        flat = per_image.reshape(-1, c)
        bank = flat[greedy_coreset(flat, per_size * n, seed=seed)]
        return cls(backbone=extractor.name, input_size=extractor.input_size, bank=bank.astype(np.float32),
                   threshold=float(max(loo)), train_scores=loo, n_train=n, grid=(h, w))

    # ------------------------------------------------------------ score
    def score(self, image: Image.Image, extractor: Extractor) -> Score:
        feats = extractor([image])[0]
        h, w, c = feats.shape
        d = nearest_distance(feats.reshape(-1, c), self.bank).reshape(h, w)
        W, H = image.size
        zoom = (H / h, W / w)
        heat = ndimage.zoom(d, zoom, order=1)[:H, :W]
        if heat.shape != (H, W):  # pad by edge values if rounding left it short
            heat = np.pad(heat, ((0, H - heat.shape[0]), (0, W - heat.shape[1])), mode="edge")
        heat = ndimage.gaussian_filter(heat, self.sigma * max(H, W) / self.input_size)
        return Score(score=float(d.max()), patch_map=d, heatmap=heat.astype(np.float32))

    # -------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format": FORMAT_VERSION, "backbone": self.backbone, "input_size": self.input_size,
                "threshold": self.threshold, "train_scores": self.train_scores, "n_train": self.n_train,
                "grid": list(self.grid), "sigma": self.sigma}
        np.savez_compressed(path, bank=self.bank, meta=np.array(json.dumps(meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PatchCoreModel":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            bank = z["bank"]
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported model format {meta.get('format')}")
        return cls(backbone=meta["backbone"], input_size=meta["input_size"], bank=bank,
                   threshold=meta["threshold"], train_scores=meta["train_scores"], n_train=meta["n_train"],
                   grid=tuple(meta["grid"]), sigma=meta["sigma"])
