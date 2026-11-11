"""Evaluate defect-inspector on the synthetic set (or any folder with the same layout).

    python scripts/evaluate.py                          # anomaly scores and heatmaps only
    python scripts/evaluate.py --describe               # also label regions with the vision model (MS_BACKEND)
    python scripts/evaluate.py --out eval.json --md eval.md

For each surface a model is fitted on ``train/good`` and every ``test`` image is scored.

- image AUROC: how well the image score separates good from defective images (1.0 = perfectly).
- pixel AUROC: how well heatmap values separate defect pixels (ground-truth masks) from the rest.
- flagged at threshold: share of defective / good images whose score is above the fitted threshold.
- top region hits: share of flagged defective images whose top region overlaps the true defect.
- label accuracy (with --describe): share of hit regions whose label matches the defect type.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

from defect_inspector.config import Settings
from defect_inspector.pipeline import fit, inspect

ROOT = Path(__file__).resolve().parent.parent


def auroc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "samples"))
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--out", help="write per-image results and metrics as JSON")
    ap.add_argument("--md", help="write the metrics table as Markdown")
    args = ap.parse_args()
    settings = Settings.from_env()
    vision = None
    if args.describe:
        from defect_inspector.describe import get_vision_model

        vision = get_vision_model(settings)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    table, per_image = [], []
    for surface, entry in manifest["surfaces"].items():
        t0 = time.time()
        model = fit(data / surface / "train" / "good", settings)
        fit_s = time.time() - t0
        ys, ss, pix_y, pix_s, times = [], [], [], [], []
        hits = labelled = correct = flagged_bad = flagged_good = n_bad = n_good = 0
        for item in entry["test"]:
            t1 = time.time()
            out = inspect(data / item["image"], model, vision=vision, max_described=1)
            times.append(time.time() - t1)
            r = out.result
            bad = item["label"] != "good"
            mask = (np.asarray(Image.open(data / item["mask"])) > 0) if item["mask"] else np.zeros(out.heatmap.shape, bool)
            ys.append(bad)
            ss.append(r.raw_score)
            pix_y.append(mask.ravel())
            pix_s.append(out.heatmap.ravel())
            hit = None
            if bad:
                n_bad += 1
                flagged_bad += r.defect
                if r.defect and r.regions:
                    x0, y0, x1, y1 = r.regions[0].bbox
                    hit = bool(mask[y0:y1, x0:x1].any())
                    hits += hit
                    if vision is not None and hit:
                        labelled += 1
                        correct += r.regions[0].label == item["label"]
            else:
                n_good += 1
                flagged_good += r.defect
            per_image.append({"surface": surface, "image": item["image"], "truth": item["label"],
                              "score": r.score, "defect": r.defect, "top_region_hit": hit,
                              "label": r.regions[0].label if r.regions else None,
                              "description": r.regions[0].description if r.regions else None})
        row = {
            "surface": surface, "train": model.n_train, "test_good": n_good, "test_defect": n_bad,
            "image_auroc": round(auroc(ys, ss), 3),
            "pixel_auroc": round(auroc(np.concatenate(pix_y), np.concatenate(pix_s)), 3),
            "flagged_defect": f"{flagged_bad}/{n_bad}", "flagged_good": f"{flagged_good}/{n_good}",
            "top_region_hits": f"{hits}/{flagged_bad}",
            "fit_s": round(fit_s, 1), "s_per_image": round(float(np.mean(times)), 2),
        }
        if vision is not None:
            row["label_accuracy"] = f"{correct}/{labelled}"
        table.append(row)
        print(json.dumps(row), file=sys.stderr)
    cols = list(table[0].keys())
    md = [f"Backbone `{settings.backbone}`" + (f", describer `{vision.name}` ({settings.backend})" if vision else ""),
          "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in table]
    print("\n".join(md))
    if args.md:
        Path(args.md).write_text("\n".join(md) + "\n")
    if args.out:
        Path(args.out).write_text(json.dumps({"backbone": settings.backbone, "describer": getattr(vision, "name", None),
                                              "metrics": table, "images": per_image}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
