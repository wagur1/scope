"""Task-relevant mask prediction and caching (MATH.md sections 2-3).

Masks are computed ONCE per sequence and cached to masks/{seq}.npz; the SAME
file must feed every arm (pre-editors AND ROI delta-QP) — council R3-R1.

Two sources:
  * YOLOMaskProvider  — public detector boxes (rectangular union masks).
    Requires `ultralytics` (guarded import). TEST-TIME SOURCE for headline
    results; no GT leakage.
  * GTBoxMaskProvider — ground-truth COCO boxes. DIAGNOSTIC ONLY (upper
    bound); never allowed in headline tables (PROTOCOL.md threats #4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def boxes_to_mask(boxes: "np.ndarray", hw: tuple[int, int]) -> "np.ndarray":
    """Union of [x1, y1, x2, y2] boxes -> float32 mask (H, W) in {0,1}."""
    h, w = hw
    m = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), w), min(int(y2), h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1.0
    return m


class YOLOMaskProvider:
    """Rectangular union masks from an ultralytics YOLO model (guarded import)."""

    def __init__(self, weights: str, conf: float = 0.25):
        try:
            from ultralytics import YOLO  # noqa: import guard
        except ImportError as exc:
            raise RuntimeError(
                "YOLOMaskProvider needs `pip install ultralytics`, or supply masks "
                "via a different provider (see PROTOCOL.md)."
            ) from exc
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, frames_dir: Path) -> dict[int, "np.ndarray"]:
        """Predict masks for %04d.png frames; returns {t: (H,W) mask}."""
        out: dict[int, "np.ndarray"] = {}
        for i, f in enumerate(sorted(frames_dir.glob("*.png"))):
            r = self.model.predict(str(f), conf=self.conf, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            out[i] = boxes_to_mask(boxes, r.orig_shape)
        return out


class GTBoxMaskProvider:
    """Diagnostic masks from a COCO-format annotation file (upper bound only).

    COCO bboxes are [x, y, w, h] — converted to [x1, y1, x2, y2] here.
    """

    def __init__(self, coco_json: str | Path):
        self.anns = json.loads(Path(coco_json).read_text())

    def __call__(self, frames_dir: Path, image_ids: list[int]) -> dict[int, "np.ndarray"]:
        by_img: dict[int, list[list[float]]] = {}
        for a in self.anns["annotations"]:
            x, y, bw, bh = a["bbox"]  # COCO format: x, y, w, h
            by_img.setdefault(a["image_id"], []).append([x, y, x + bw, y + bh])
        out: dict[int, "np.ndarray"] = {}
        first = sorted(frames_dir.glob("*.png"))[0]
        from PIL import Image

        w, h = Image.open(first).size
        for t, img_id in enumerate(image_ids):
            out[t] = boxes_to_mask(np.array(by_img.get(img_id, [])), (h, w))
        return out


def save_masks(path: Path, masks: dict[int, "np.ndarray"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"t{i:04d}": m for i, m in sorted(masks.items())})


def load_masks(path: Path) -> dict[int, "np.ndarray"]:
    with np.load(path) as data:
        return {int(k[1:]): data[k] for k in data.files}


class LazyMasks:
    """Per-frame access into a masks npz without materializing the sequence.

    experiment.py needs masks frame-by-frame (streaming policy); a full
    dict of 1080p float32 masks is ~8 MB/frame — 50 GB for MOT17-02.
    """

    def __init__(self, path: Path, n_expected: int | None = None):
        self.path = Path(path)
        with np.load(self.path) as data:
            keys = sorted(int(k[1:]) for k in data.files)
        if not keys:
            raise ValueError(f"{self.path}: empty mask archive")
        if n_expected is not None and len(keys) != n_expected:
            raise ValueError(
                f"{self.path}: {len(keys)} masks != {n_expected} frames — "
                "mask cache and frame directory are out of sync"
            )
        self._indices = set(keys)

    def __getitem__(self, t: int) -> "np.ndarray":
        if t not in self._indices:
            raise KeyError(f"mask for frame {t} not in {self.path}")
        with np.load(self.path) as data:
            return data[f"t{t:04d}"]
