"""Patch feature extractors for PatchCore.

``timm/wide_resnet50_2.tv_in1k`` (default): layer2 and layer3 features, each
averaged over a 3x3 neighbourhood, layer3 upsampled to the layer2 grid, then
concatenated and reduced to ``dim`` channels by adaptive average pooling, as in
the PatchCore paper. A 256 px image gives a 32 x 32 grid of patch vectors.

``facebook/dinov2-*``: patch tokens from the last hidden layer (needs the
``hf`` extra). A 252 px input gives an 18 x 18 grid.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Extractor(Protocol):
    name: str
    input_size: int

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        """Return features of shape (N, H, W, C) as float32."""
        ...


def to_tensor(images: list[Image.Image], size: int):
    import torch

    arrs = []
    for im in images:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        a = (np.asarray(im, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        arrs.append(a.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(arrs))


class TimmPatchFeatures:
    """Mid-level CNN features from a timm backbone (any model with ``features_only`` support)."""

    def __init__(self, name: str, input_size: int = 256, dim: int = 1024, pretrained: bool = True,
                 layers: tuple[int, ...] = (2, 3)):
        import timm
        import torch

        self._torch = torch
        self.name = name
        self.input_size = input_size
        self.dim = dim
        model_name = name.removeprefix("hf-hub:").removeprefix("hf_hub:")
        if "/" in model_name and not model_name.startswith("hf-hub:"):
            model_name = f"hf-hub:{model_name}"
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                       out_indices=layers).eval()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        F = torch.nn.functional
        with torch.no_grad():
            feats = self.model(to_tensor(images, self.input_size))
            feats = [F.avg_pool2d(f, 3, stride=1, padding=1) for f in feats]
            h, w = feats[0].shape[-2:]
            feats = [feats[0]] + [F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
                                  for f in feats[1:]]
            x = torch.cat(feats, dim=1)  # N, C, H, W
            n, c = x.shape[:2]
            x = x.permute(0, 2, 3, 1).reshape(-1, 1, c)
            if c != self.dim:
                x = F.adaptive_avg_pool1d(x, self.dim)
            return x.reshape(n, h, w, -1).numpy().astype(np.float32)


class DinoPatchFeatures:
    """Patch tokens from a DINOv2 model loaded with ``transformers``."""

    def __init__(self, name: str, input_size: int = 252):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("DINOv2 backbones need the 'hf' extra: pip install -e '.[hf]'") from exc
        self._torch = torch
        self.name = name
        self.input_size = input_size - input_size % 14
        self.model = AutoModel.from_pretrained(name).eval()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            out = self.model(pixel_values=to_tensor(images, self.input_size))
            tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
            side = self.input_size // 14
            x = tokens.reshape(tokens.shape[0], side, side, -1)
            x = torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), 3, stride=1, padding=1).permute(0, 2, 3, 1)
            return x.numpy().astype(np.float32)


_CACHE: dict[tuple, Extractor] = {}


def get_extractor(name: str, input_size: int = 256) -> Extractor:
    key = (name, input_size)
    if key not in _CACHE:
        if "dinov2" in name.lower():
            _CACHE[key] = DinoPatchFeatures(name, input_size)
        else:
            _CACHE[key] = TimmPatchFeatures(name, input_size)
    return _CACHE[key]
