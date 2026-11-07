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


def detect_frames(frames_dir: Path, weights: str, conf: float = 0.001,
                  classes: list[int] | None = None) -> list[dict]:
    """Run detection over %04d.png frames. Returns per-frame COCO-style dicts.

    Backend dispatch (2026-11: aligned with VCM literature per reviewer note —
    recent papers standardize on YOLOv5 (Dual-Region ITC-CSCC'25, Shindo
    CCNC'24) and Faster R-CNN (MPEG VCM CTC anchor; SCOPE, Lu TCSVT'24)):
      * weights startswith "yolo"  -> ultralytics (yolov5su.pt / yolov8n.pt)
      * weights startswith "frcnn" -> torchvision fasterrcnn_resnet50_fpn_v2
        (pretrained, zero-install; CTC-exact Detectron2 X101 is the
         full-resolution post-gate upgrade)

    category_id conventions: YOLO emits cls+1; torchvision labels are already
    COCO ids (person = 1). classes filters in each backend's native ids:
    YOLO [0] = person, torchvision [1] = person.
    conf=0.001 for evaluation recall (standard COCO protocol).
    """
    if weights.startswith("frcnn"):
        return _detect_torchvision(frames_dir, conf, classes)
    return _detect_ultralytics(frames_dir, weights, conf, classes)


def _detect_torchvision(frames_dir: Path, conf: float,
                        classes: list[int] | None) -> list[dict]:
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # review fix (Astra): the constructor's internal box_score_thresh=0.05
    # discarded detections BEFORE our conf=0.001 filter — pass conf through
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=conf,
    )
    model.to(device).eval()
    tfm = torchvision.transforms.functional

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        # review fix (Astra): missing decoded dirs produced empty detection
        # lists scored as quality=0 — an infrastructure failure, not a result
        raise FileNotFoundError(f"no decoded frames in {frames_dir}")
    results: list[dict] = []
    for t, f in enumerate(frames):
        img = tfm.to_tensor(Image.open(f).convert("RGB")).to(device)
        with torch.no_grad():
            out = model([img])[0]
        keep = out["scores"] >= conf
        labels = out["labels"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        boxes = out["boxes"][keep].cpu().numpy()
        for b, s, l in zip(boxes, scores, labels):
            if classes is not None and int(l) not in classes:
                continue
            x1, y1, x2, y2 = b
            results.append({
                "image_id": t,
                "category_id": int(l),  # torchvision labels are COCO ids
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
    return results


def _detect_ultralytics(frames_dir: Path, weights: str, conf: float,
                        classes: list[int] | None) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("detection eval needs `pip install ultralytics`") from exc

    model = YOLO(weights)
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no decoded frames in {frames_dir}")
    results: list[dict] = []
    for t, f in enumerate(frames):
        r = model.predict(str(f), conf=conf, verbose=False, classes=classes)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        cls = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        for b, s, c in zip(boxes, scores, cls):
            x1, y1, x2, y2 = b
            results.append({
                "image_id": t,
                "category_id": int(c) + 1,  # YOLO cls 0-based -> COCO id
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
    return results


def mAP(detections: list[dict], gt_json: str | Path, img_ids: list[int],
        use_cats: bool = True) -> float:
    """COCO mAP@[.5:.95] over the given image ids (guarded pycocotools).

    use_cats=False runs class-agnostic evaluation (COCOeval useCats=0) —
    right for synthetic data where the scorer may label the planted object
    with any class. Loud-failure policy (code-review fix): id/category
    mismatches make COCOeval silently return 0.0 — asserted here instead.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("scoring needs `pip install pycocotools`") from exc

    gt = COCO(str(gt_json))
    gt_ids = set(gt.getImgIds())
    # decoded-frame count must match GT images exactly (review fix): partial
    # directories scored against full GT silently corrupt RD curves
    bad_imgs = {d["image_id"] for d in detections} - gt_ids
    if bad_imgs:
        raise ValueError(
            f"detection image_ids not present in GT ({sorted(bad_imgs)[:5]}...): "
            "frame index <-> GT image_id mapping is wrong — fix before scoring, "
            "silent-zero mAP is not acceptable"
        )
    expected_ids = set(img_ids) & gt_ids
    if expected_ids != gt_ids:
        raise ValueError(f"GT image ids missing from the evaluation set: "
                         f"{sorted(gt_ids - expected_ids)[:5]}... — partial decode "
                         "would corrupt the RD curve")
    gt_cats = set(gt.getCatIds())
    bad_cats = {d["category_id"] for d in detections} - gt_cats
    if bad_cats and use_cats:
        raise ValueError(
            f"detection category_ids not in GT ({sorted(bad_cats)[:5]}...): "
            "VERIFY the class-id mapping (detect.py adds +1 to YOLO cls); "
            "or run class-agnostic (use_cats=False) for synthetic GT"
        )
    if not detections:
        return 0.0
    dt = gt.loadRes(detections)
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.useCats = 1 if use_cats else 0
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
