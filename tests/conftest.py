from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defect_inspector import features

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class FakeExtractor:
    """Stands in for the CNN: 8 x 8 pixel blocks, mean and std per colour channel, on a 32 x 32 grid."""

    name = "fake-backbone"
    input_size = 256

    def __call__(self, images):
        out = []
        for im in images:
            a = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
            blocks = a.reshape(32, 8, 32, 8, 3).transpose(0, 2, 1, 3, 4).reshape(32, 32, 64, 3)
            out.append(np.concatenate([blocks.mean(2), blocks.std(2) * 4], axis=-1))
        return np.stack(out).astype(np.float32)


class FakeVision:
    name = "fake-vlm"

    def __init__(self, reply='{"label": "scratch", "description": "a thin light line"}'):
        self.reply = reply
        self.calls = 0

    def ask(self, image, prompt):
        self.calls += 1
        assert isinstance(image, Image.Image) and "label" in prompt
        return self.reply


@pytest.fixture(autouse=True)
def fake_backbone(monkeypatch, tmp_path):
    """Every test uses the fake extractor under the name 'fake-backbone'; no model downloads."""
    monkeypatch.setitem(features._CACHE, ("fake-backbone", 256), FakeExtractor())
    monkeypatch.setenv("MS_BACKBONE", "fake-backbone")
    monkeypatch.setenv("MS_CACHE_DIR", str(tmp_path / "cache"))
    # empty values (not unset) so a local .env cannot fill them in
    for var in ("MS_BACKEND", "MS_API_BASE", "MS_API_KEY", "MS_MODEL"):
        monkeypatch.setenv(var, "")
    return FakeExtractor()


@pytest.fixture
def samples() -> Path:
    return SAMPLES


@pytest.fixture
def fake_vision():
    return FakeVision
