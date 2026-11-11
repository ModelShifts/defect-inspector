import numpy as np
import pytest
from PIL import Image

from defect_inspector.patchcore import PatchCoreModel, greedy_coreset, nearest_distance


def _imgs(folder):
    return [Image.open(p).convert("RGB") for p in sorted(folder.glob("*.png"))]


def test_nearest_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    q, b = rng.standard_normal((50, 8)), rng.standard_normal((70, 8))
    want = np.sqrt(((q[:, None] - b[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(nearest_distance(q, b, chunk=16), want, atol=1e-4)


def test_greedy_coreset_spreads_out():
    rng = np.random.default_rng(1)
    pts = np.concatenate([rng.normal(0, 0.1, (100, 4)), 10 + rng.normal(0, 0.1, (3, 4))]).astype(np.float32)
    idx = greedy_coreset(pts, 4, proj_dim=2)
    assert len(set(idx.tolist())) == 4
    assert (idx >= 100).sum() >= 1  # the far cluster is always picked


def test_fit_score_save_load(samples, fake_backbone, tmp_path):
    goods = _imgs(samples / "metal" / "train" / "good")
    model = PatchCoreModel.fit(goods, fake_backbone, coreset_ratio=0.25)
    assert model.n_train == 8 and model.bank.shape[1] == 6 and len(model.train_scores) == 8
    assert model.threshold == pytest.approx(max(model.train_scores))
    good = model.score(Image.open(samples / "metal/test/good/000.png"), fake_backbone)
    bad = model.score(Image.open(samples / "metal/test/spot/000.png"), fake_backbone)
    assert bad.score > good.score
    assert bad.heatmap.shape == (256, 256) and bad.patch_map.shape == (32, 32)
    path = model.save(tmp_path / "m")
    assert path.suffix == ".npz"
    again = PatchCoreModel.load(path)
    assert again.threshold == model.threshold and np.array_equal(again.bank, model.bank)
    assert again.score(Image.open(samples / "metal/test/spot/000.png"), fake_backbone).score == pytest.approx(bad.score)


def test_fit_needs_two_images(fake_backbone, samples):
    with pytest.raises(ValueError):
        PatchCoreModel.fit(_imgs(samples / "metal" / "train" / "good")[:1], fake_backbone)


def test_timm_extractor_shapes_without_download():
    from defect_inspector.features import TimmPatchFeatures

    ex = TimmPatchFeatures("resnet18", input_size=128, dim=64, pretrained=False)
    out = ex([Image.new("RGB", (100, 80), (120, 130, 140))] * 2)
    assert out.shape == (2, 16, 16, 64) and out.dtype == np.float32
