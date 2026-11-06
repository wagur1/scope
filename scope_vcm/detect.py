"""Detection evaluation: run a frozen detector on decoded frames, score mAP.

Task-model policy (council R2): the SAME frozen detector scores every arm.
`ultralytics` YOLO is the default runner; pycocotools is the scorer.
Both imports are guarded so the rest of the pipeline installs/runs without them.

GT must be provided as a COCO-format json whose image order matches the
frame numbering of each sequence (config field `gt_json`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def detect_frames(frames_dir: Path, weights: str, conf: float = 0.001) -> list[dict]:
    """Run detection over %04d.png frames. Returns per-frame COCO-style dicts.

    conf=0.001 for evaluation recall (standard COCO protocol), distinct from
    the mask-provider threshold.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("detection eval needs `pip install ultralytics`") from exc

    model = YOLO(weights)
    results: list[dict] = []
    for t, f in enumerate(sorted(frames_dir.glob("*.png"))):
        r = model.predict(str(f), conf=conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        cls = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        for b, s, c in zip(boxes, scores, cls):
            x1, y1, x2, y2 = b
            results.append({
                "image_id": t,
                "category_id": int(c) + 1,  # VERIFY class-id mapping against gt_json
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
    return results


def mAP(detections: list[dict], gt_json: str | Path, img_ids: list[int]) -> float:
    """COCO mAP@[.5:.95] over the given image ids (guarded pycocotools).

    Loud-failure policy (code-review fix): id/category mismatches make
    COCOeval silently return 0.0 — asserted here instead.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("scoring needs `pip install pycocotools`") from exc

    gt = COCO(str(gt_json))
    gt_ids = set(gt.getImgIds())
    bad_imgs = {d["image_id"] for d in detections} - gt_ids
    if bad_imgs:
        raise ValueError(
            f"detection image_ids not present in GT ({sorted(bad_imgs)[:5]}...): "
            "frame index <-> GT image_id mapping is wrong — fix before scoring, "
            "silent-zero mAP is not acceptable"
        )
    gt_cats = set(gt.getCatIds())
    bad_cats = {d["category_id"] for d in detections} - gt_cats
    if bad_cats:
        raise ValueError(
            f"detection category_ids not in GT ({sorted(bad_cats)[:5]}...): "
            "VERIFY the class-id mapping (detect.py adds +1 to YOLO cls)"
        )
    if not detections:
        return 0.0
    dt = gt.loadRes(detections)
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.imgIds = sorted(set(img_ids) & gt_ids)
    e.evaluate()
    e.accumulate()
    e.summarize()
    return float(e.stats[0])


def ceiling_mAP(frames_dir: Path, gt_json: str | Path, weights: str,
                img_ids: list[int], img_id_map: dict[int, int] | None = None) -> float:
    """mAP at negligible compression: decode-free, run on the source frames.

    For pre-edited sequences this is m*_E (MATH.md section 4); pass the
    edited frame dir. For the anchor this is m*_anchor: pass the original.
    img_id_map remaps frame index -> GT image_id when ids are not 0..T-1.
    """
    dets = detect_frames(frames_dir, weights)
    if img_id_map is not None:
        for d in dets:
            d["image_id"] = img_id_map[d["image_id"]]
    return mAP(dets, gt_json, img_ids)
