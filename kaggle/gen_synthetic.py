#!/usr/bin/env python
"""Synthetic E1 data generator for the Kaggle smoke run.

Generates sequences with known GT (single moving object class) + a COCO-format
GT json + an e1 config. Declared purpose: PIPELINE VALIDATION on a fresh
environment (encoder acceptance, BD machinery, gates) — NOT thesis evidence;
the scorer is a real YOLO but content/GT is synthetic, so quality axes are
meaningful only mechanically (see kaggle/kernel_driver.py notes).

Per-account seed differs -> the cross-account bitrate agreement of the SAME
seed block doubles as the encoder-determinism check (README threat 5).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.masks import save_masks  # noqa: E402

WORK = Path(os.environ.get("SYNTH_WORK", "/kaggle/working"))


def make_sequence(rng: np.random.Generator, sid: str, n: int, hw: tuple[int, int],
                  noise_sigma: float, texture: bool) -> tuple[Path, list[dict]]:
    """Moving box on noisy (optionally textured) background. Returns frames dir + GT entries."""
    h, w = hw
    frames_dir = WORK / "synth" / sid / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gt = []
    base = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    if texture:  # block-textured background: harder to compress than pure noise
        for y in range(0, h, 16):
            for x in range(0, w, 16):
                base[y:y + 16, x:x + 16] = rng.integers(40, 220, (3,), dtype=np.uint8)
    for t in range(n):
        img = base.copy()
        if noise_sigma > 0:
            noise = rng.normal(0, noise_sigma, img.shape)
            img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        x0, y0 = 20 + t * 2, 20 + (t * 3) % max(h - 60, 1)
        bw, bh = 40, 40
        img[y0:y0 + bh, x0:x0 + bw] = [255, 40, 40]  # red box
        Image.fromarray(img).save(frames_dir / f"{t:04d}.png")
        gt.append({"image_id": t, "category_id": 1,
                   "bbox": [float(x0), float(y0), float(bw), float(bh)],
                   "area": float(bw * bh), "iscrowd": 0})
    return frames_dir, gt


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)

    # 8 small + 4 larger sequences, varied noise/texture (content grid on one axis
    # of the failure-boundary taxonomy — synthetic stand-ins for now)
    specs = [f"s{i}" for i in range(8)] + [f"L{i}" for i in range(4)]
    seq_cfg, all_gt, categories = [], {}, [{"id": 1, "name": "box"}]
    images_meta = []
    for i, sid in enumerate(specs):
        small = i < 8
        n = 48 if small else 64
        hw = (128, 160) if small else (180, 320)
        sigma = [0.0, 10.0, 25.0, 40.0][i % 4]
        textured = i % 2 == 1
        frames_dir, gt = make_sequence(rng, sid, n, hw, sigma, textured)
        for g in gt:
            images_meta.append({"id": g["image_id"], "width": hw[1], "height": hw[0],
                                "file_name": f"{sid}/{g['image_id']:04d}.png"})
        # per-sequence GT json with global ids shifted (image_id must be unique per file)
        shift = i * 1000
        for g in gt:
            g["image_id"] += shift
        all_gt[sid] = {"images": images_meta[-n:], "annotations": gt, "categories": categories}
        seq_cfg.append({"id": sid, "frames": str(frames_dir)})

        masks = {}
        for g in gt:
            m = np.zeros((hw[0], hw[1]), dtype=np.float32)
            x, y, bw, bh = [int(v) for v in g["bbox"]]
            m[y:y + bh, x:x + bw] = 1.0
            masks[g["image_id"] - shift] = m
        save_masks(WORK / "synth_masks" / f"{sid}.npz", masks)

    # GT: one file per sequence (eval_e1 scores per-sequence image-id spaces)
    gt_dir = WORK / "synth_gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for sid, payload in all_gt.items():
        (gt_dir / f"{sid}.json").write_text(json.dumps(payload))

    config = {
        "workdir": str(WORK / "work_e1"),
        "encoder": "x265",
        "preset": "faster",
        "fps": 30.0,
        "qp_grid": [22, 27, 32, 37, 42],
        "roi_deltas": [8, 16],
        "blur_sigma": 8.0,
        "arms": {"A0_anchor": True, "A1_roi": True, "A2_preedit_fb": True,
                 "A2c_preedit_blurall": True, "A4_preedit_fb_roi": True},
        "sequences": seq_cfg,
        "gt_dir": str(gt_dir),  # per-sequence GT jsons (image ids collide across sequences)
        "masks_dir": str(WORK / "synth_masks"),
        "agnostic": True,  # synthetic GT: class-agnostic scoring (useCats=0)
        "seed": seed,
    }
    (WORK / "e1_synth.yaml").write_text(json.dumps(config, indent=2))
    print(f"generated {len(specs)} sequences, seed={seed}, config -> e1_synth.yaml")


if __name__ == "__main__":
    main()
