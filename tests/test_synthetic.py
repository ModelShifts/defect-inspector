import json

import numpy as np
from PIL import Image

from defect_inspector.synthetic import DEFECTS, SURFACES, make_dataset


def test_dataset_layout_and_masks(tmp_path):
    m = make_dataset(tmp_path, n_train=2, n_test_good=1, n_test_defect=1, size=96)
    assert set(m["surfaces"]) == set(SURFACES)
    for surface, entry in m["surfaces"].items():
        assert len(entry["train"]) == 2
        labels = [t["label"] for t in entry["test"]]
        assert labels.count("good") == 1 and all(d in labels for d in DEFECTS)
        for t in entry["test"]:
            img = Image.open(tmp_path / t["image"])
            assert img.size == (96, 96) and img.mode == "RGB"
            if t["mask"]:
                mask = np.asarray(Image.open(tmp_path / t["mask"]))
                assert mask.shape == (96, 96) and mask.max() == 255 and (mask > 0).mean() < 0.3
    assert json.loads((tmp_path / "manifest.json").read_text())["seed"] == 7


def test_same_seed_same_images(tmp_path):
    a = make_dataset(tmp_path / "a", 1, 0, 1, surfaces=("stone",), size=64)
    make_dataset(tmp_path / "b", 1, 0, 1, surfaces=("stone",), size=64)
    for item in a["surfaces"]["stone"]["test"]:
        x = np.asarray(Image.open(tmp_path / "a" / item["image"]))
        y = np.asarray(Image.open(tmp_path / "b" / item["image"]))
        assert (x == y).all()


def test_committed_samples_match_generator(samples, tmp_path):
    """samples/ is what scripts/make_samples.py writes (allowing for 1-level rounding differences)."""
    make_dataset(tmp_path, n_train=8, n_test_good=6, n_test_defect=4)
    manifest = json.loads((samples / "manifest.json").read_text())
    for surface in SURFACES:
        for rel in [manifest["surfaces"][surface]["train"][0], manifest["surfaces"][surface]["test"][-1]["image"]]:
            got = np.asarray(Image.open(tmp_path / rel), dtype=np.int16)
            want = np.asarray(Image.open(samples / rel), dtype=np.int16)
            assert np.abs(got - want).max() <= 1, rel
