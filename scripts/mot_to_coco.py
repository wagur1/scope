#!/usr/bin/env python
"""Convert MOT17-format GT to COCO json for E1 detection eval.

MOT GT format (gt/gt.txt): frame,id,x,y,w,h,conf,class,visibility
E1 uses: class==1 (pedestrian), conf>=0.55, visibility>=0.55, min box size.
One COCO json per sequence (per-sequence image ids 0..T-1 — matches the
per-sequence eval mode, no cross-sequence collision).

Usage:
  python scripts/mot_to_coco.py <seq_dir> [--out gt_coco.json]
  # <seq_dir> contains img1/*.png (or *.jpg) and gt/gt.txt
"""
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--min-vis", type=float, default=0.55)
    ap.add_argument("--min-size", type=int, default=16)
    args = ap.parse_args()

    seq = Path(args.seq_dir)
    frames = sorted((seq / "img1").glob("*.png")) or sorted((seq / "img1").glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames under {seq}/img1")
    gt_file = seq / "gt" / "gt.txt"
    if not gt_file.exists():
        raise SystemExit(f"missing {gt_file}")

    from PIL import Image
    w0, h0 = Image.open(frames[0]).size

    anns, ann_id = [], 0
    for line in gt_file.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 9:
            continue
        frame, _tid, x, y, w, h, conf, cls, vis = parts[:9]
        if int(cls) != 1 or float(conf) < args.min_conf or float(vis) < args.min_vis:
            continue
        x, y, w, h = float(x), float(y), float(w), float(h)
        if w < args.min_size or h < args.min_size:
            continue
        t = int(frame) - 1  # MOT frames are 1-based
        if t >= len(frames):
            continue
        anns.append({"id": ann_id, "image_id": t, "category_id": 1,
                     "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                     "attributes": {"visibility": float(vis)}})
        ann_id += 1

    payload = {
        "images": [{"id": t, "width": w0, "height": h0,
                    "file_name": frames[t].name} for t in range(len(frames))],
        "annotations": anns,
        "categories": [{"id": 1, "name": "pedestrian"}],
    }
    out = Path(args.out) if args.out else seq / "gt_coco.json"
    out.write_text(json.dumps(payload))
    print(f"{seq.name}: {len(anns)} boxes / {len(frames)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
