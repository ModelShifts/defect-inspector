import csv
import json

import pytest

from defect_inspector import cli
from defect_inspector.config import Settings
from defect_inspector.pipeline import JSONCache, fit, inspect


def test_cli_fit_inspect_batch(samples, tmp_path, capsys):
    model = tmp_path / "metal.npz"
    assert cli.main(["fit", str(samples / "metal/train/good"), "--out", str(model)]) == 0
    assert model.exists()
    heat = tmp_path / "out" / "heat.png"
    assert cli.main(["inspect", str(samples / "metal/test/scratch/000.png"), "--model", str(model),
                     "--heatmap", str(heat), "--json", str(tmp_path / "r.json")]) == 0
    result = json.loads((tmp_path / "r.json").read_text())
    assert result["defect"] is True and result["score"] > 1 and result["regions"]
    assert heat.exists() and result["heatmap_png"] == str(heat)
    capsys.readouterr()
    out_csv = tmp_path / "batch.csv"
    assert cli.main(["batch", str(samples / "metal/test"), "--recursive", "--model", str(model),
                     "--out", str(out_csv), "--heatmaps", str(tmp_path / "heat")]) == 0
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 18 and set(rows[0]) >= {"file", "defect", "score", "top_bbox", "top_label"}
    assert sum(int(r["defect"]) for r in rows if not r["file"].startswith("good/")) >= 10
    assert (tmp_path / "heat" / "scratch" / "000.heatmap.png").exists()
    out_json = tmp_path / "batch.json"
    assert cli.main(["batch", str(samples / "metal/test/good"), "--model", str(model), "--out", str(out_json)]) == 0
    assert len(json.loads(out_json.read_text())) == 6


def test_cli_errors(samples, tmp_path, monkeypatch):
    assert cli.main(["inspect", str(samples / "metal/test/good/000.png"), "--model", str(tmp_path / "nope.npz")]) == 2
    model = tmp_path / "m.npz"
    cli.main(["fit", str(samples / "metal/train/good"), "--out", str(model)])
    monkeypatch.setenv("MS_BACKEND", "api")  # no MS_API_BASE / MS_MODEL: clear error, exit 2
    assert cli.main(["inspect", str(samples / "metal/test/dent/000.png"), "--model", str(model), "--describe"]) == 2


def test_describe_uses_vision_model_and_cache(samples, fake_vision, tmp_path):
    model = fit(samples / "fabric/train/good", Settings.from_env())
    vision = fake_vision()
    cache = JSONCache(tmp_path / "describe")
    out = inspect(samples / "fabric/test/scratch/000.png", model, vision=vision, describe_cache=cache)
    assert out.result.regions and out.result.regions[0].label == "scratch"
    assert out.result.describer == "fake-vlm"
    calls = vision.calls
    inspect(samples / "fabric/test/scratch/000.png", model, vision=vision, describe_cache=cache)
    assert vision.calls == calls  # second run is served from the cache


def test_good_image_has_no_regions(samples):
    model = fit(samples / "fabric/train/good", Settings.from_env())
    out = inspect(samples / "fabric/train/good/000.png", model)
    assert out.result.defect is False and out.result.regions == []


def test_api_backend_sends_image(monkeypatch):
    from defect_inspector import describe

    sent = {}

    class FakeCompletions:
        def create(self, **kw):
            sent.update(kw)
            msg = type("M", (), {"content": '{"label": "dent", "description": "round dip"}'})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    class FakeClient:
        def __init__(self, **kw):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    s = Settings(backend="api", api_base="http://x/v1", model="vlm")
    v = describe.APIVision(s)
    from PIL import Image

    assert describe.describe(Image.new("RGB", (32, 32)), v) == ("dent", "round dip")
    content = sent["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,") and sent["temperature"] == 0


def test_web_app_builds():
    pytest.importorskip("gradio")
    from defect_inspector.webapp import build_app, test_images

    assert len(test_images("metal")) == 18
    assert build_app(Settings.from_env()) is not None
