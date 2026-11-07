#!/usr/bin/env python
"""Predict task-relevant masks for every configured sequence and cache them.

The ONE shared mask file per sequence feeds every arm (pre-editors AND the
ROI delta-QP maps) — council R3-R1. Run this before `run_e1.py`.

Usage:
  python scripts/make_masks.py <config> [--weights yolov8n.pt] [--conf 0.25]
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.masks import YOLOMaskProvider, save_masks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="detector checkpoint — pinned in the run manifest")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["workdir"]).resolve()
    provider = YOLOMaskProvider(args.weights, conf=args.conf)

    for seq in cfg["sequences"]:
        sid = seq["id"]
        frames_dir = Path(seq["frames"]).resolve()
        out = root / "masks" / f"{sid}.npz"
        if out.exists():
            logging.info("%s: masks exist, skipping (%s)", sid, out)
            continue
        logging.info("%s: predicting masks with %s ...", sid, args.weights)
        masks = provider(frames_dir)
        save_masks(out, masks)
        logging.info("%s: %d masks -> %s", sid, len(masks), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
