import numpy as np
from PIL import Image

from defect_inspector.describe import LABELS, parse_reply
from defect_inspector.models import Region
from defect_inspector.regions import crop_region, find_regions, overlay


def _heat():
    h = np.full((128, 128), 0.5, dtype=np.float32)
    h[20:30, 40:60] = 2.0
    h[90:100, 90:96] = 1.1
    return h


def test_find_regions_orders_by_peak_and_respects_sensitivity():
    regions = find_regions(_heat(), threshold=1.0, core=0.0)
    assert [r.bbox for r in regions] == [[40, 20, 60, 30], [90, 90, 96, 100]]
    assert regions[0].peak == 2.0 and regions[0].area_px == 200
    assert len(find_regions(_heat(), threshold=1.0, sensitivity=0.6, core=0.0)) == 1
    assert len(find_regions(_heat(), threshold=1.0, sensitivity=1.0, core=0.0)) == 2
    # the core cut keeps only areas near the highest value
    assert len(find_regions(_heat(), threshold=1.0)) == 1


def test_overlay_and_crop():
    img = Image.new("RGB", (128, 128), (128, 128, 128))
    out = overlay(img, _heat(), 1.0, [Region(id=1, bbox=[40, 20, 60, 30], area_px=200, peak=2.0)])
    assert out.size == (512, 512) and out.mode == "RGB"
    assert out.getpixel((200, 100)) != (128, 128, 128)  # coloured where the heat is
    crop = crop_region(img, Region(id=1, bbox=[0, 0, 10, 10], area_px=100, peak=2.0))
    assert crop.size == (384, 384)


def test_parse_reply_variants():
    assert parse_reply('{"label": "Dent", "description": "a round dip"}') == ("dent", "a round dip")
    assert parse_reply('```json\n{"label":"spot","description":"brown stain"}\n```') == ("spot", "brown stain")
    label, desc = parse_reply("I think this is a scratch running across.")
    assert label == "scratch" and "scratch" in desc
    assert parse_reply("no idea")[0] == "other"
    assert parse_reply('{"label": "hole", "description": "x"}')[0] in LABELS
