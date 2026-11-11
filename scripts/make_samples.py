"""Generate the synthetic sample set in samples/ (or another folder).

    python scripts/make_samples.py                 # samples/: 3 surfaces x (8 train + 18 test) images
    python scripts/make_samples.py --out data/big --train 16 --test-good 20 --test-defect 10 --seed 11

All images are procedural textures with injected scratches, dents and spots.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from defect_inspector.synthetic import make_dataset

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "samples"))
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test-good", type=int, default=6)
    ap.add_argument("--test-defect", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    for surface in ("metal", "fabric", "stone"):
        shutil.rmtree(out / surface, ignore_errors=True)
    m = make_dataset(out, args.train, args.test_good, args.test_defect, seed=args.seed)
    n = sum(len(v["train"]) + len(v["test"]) for v in m["surfaces"].values())
    print(f"wrote {n} images to {out}")


if __name__ == "__main__":
    main()
