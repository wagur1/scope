#!/usr/bin/env python
"""Real-data E1 kernel: MOT17 (FRCNN variants) via the mot17-coco dataset.

Design (one kernel = full 7-sequence pass, ~1.5-2h on Kaggle CPU):
  * frames: first 150 per sequence, resized to 960x540 (encode-budget smoke;
    the full-resolution full-length grid is post-gate work)
  * arms: A0 / A2 / A2c on x264 CQP (A1 waits for the ROI path — see ROI hunt)
  * scorer: frozen YOLOv8n, person class only, vs MOT GT (pedestrian), scaled
    to the resized frames
  * all pool accounts run the IDENTICAL job -> replicates double as a
    cross-machine determinism check on real encoder output

Output: out/{e1.csv, quality.csv, ceiling.csv, report.md, summary.json}
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

WORK = Path("/kaggle/working")
REPO = WORK / "scope"
# Dataset mount naming varies (slug vs title) — discover the annotations file
# dynamically instead of hardcoding /kaggle/input/<slug>/MOT17/... (round-1
# real-E1 failed on exactly this).


def find_dataset_root() -> Path:
    hits = list(Path("/kaggle/input").rglob("annotations/train.json"))
    if not hits:
        raise FileNotFoundError(
            f"annotations/train.json not under /kaggle/input: "
            f"{[str(p) for p in Path('/kaggle/input').iterdir()]}")
    return hits[0].parents[1]


DATASET = None  # resolved in main()
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN",
        "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
FRAME_LIMIT = 150
TARGET = (540, 960)  # (h, w)
QPS = [27, 32, 37]
summary: dict = {"steps": {}}



def record_gpu(summary: dict) -> None:
    """Which GPU did the API actually assign? (metadata has no type field)"""
    try:
        import torch
        if torch.cuda.is_available():
            summary["gpu"] = [torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())]
        else:
            summary["gpu"] = "none"
    except Exception as exc:
        summary["gpu"] = f"probe failed: {exc}"

def run(name: str, cmd: list, cwd=None, check: bool = True, timeout=7200) -> str:
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)
    log = WORK / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(map(str, cmd))}\n\n--- stdout ---\n"
                   f"{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    summary["steps"][name] = {"returncode": proc.returncode,
                              "seconds": round(time.time() - t0, 1)}
    print(f"[{name}] rc={proc.returncode} ({summary['steps'][name]['seconds']}s)")
    if check and proc.returncode != 0:
        raise RuntimeError(f"{name} failed — see {log}")
    return proc.stdout + proc.stderr


def prepare(seq_name: str, frame_limit: int, qps: list) -> tuple[dict, Path]:
    """Frames (resized) + per-seq COCO GT (scaled) + YOLO masks + e1 config."""
    from PIL import Image
    from scope_vcm.masks import YOLOMaskProvider, save_masks

    train = json.loads((DATASET / "annotations" / "train.json").read_text())
    id2img = {im["id"]: im for im in train["images"]}
    anns_by_img: dict = {}
    for a in train["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    root = WORK / "real_e1"
    gt_dir = root / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    masker = YOLOMaskProvider("yolov5su.pt", conf=0.25)  # YOLOv5 = VCM-literature standard
    seq_cfg = []

    for sid in [seq_name]:
        img_dir = DATASET / "train" / sid / "img1"
        if not img_dir.exists():
            raise FileNotFoundError(f"{sid}: missing {img_dir}")
        files = sorted(img_dir.glob("*.jpg"))[:frame_limit]
        out_dir = root / sid / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        gt, masks = [], []
        for t, f in enumerate(files):
            im = cv2.imread(str(f))
            oh, ow = im.shape[:2]
            sx, sy = TARGET[1] / ow, TARGET[0] / oh
            im = cv2.resize(im, (TARGET[1], TARGET[0]))
            cv2.imwrite(str(out_dir / f"{t:04d}.png"), im)
            for a in anns_by_img.get(id2img_by_file(sid, f.name), []):
                x, y, w, h = a["bbox"]
                gt.append({"id": len(gt), "image_id": t, "category_id": 1,
                           "bbox": [x * sx, y * sy, w * sx, h * sy],
                           "area": w * sx * h * sy, "iscrowd": 0})
        # masks from YOLO on the resized frames
        m = masker(out_dir)
        save_masks(root / "masks" / f"{sid}.npz", m)

        (gt_dir / f"{sid}.json").write_text(json.dumps({
            "images": [{"id": t, "width": TARGET[1], "height": TARGET[0],
                        "file_name": f"{t:04d}.png"} for t in range(len(files))],
            "annotations": gt, "categories": [{"id": 1, "name": "pedestrian"}]}))
        seq_cfg.append({"id": sid, "frames": str(out_dir)})
        print(f"prepared {sid}: {len(files)} frames, {len(gt)} GT boxes")

    cfg = {
        "workdir": str(root / "work"),
        "encoder": "x264", "preset": "veryfast", "fps": 30.0,
        "qp_grid": qps, "roi_deltas": [], "blur_sigma": 8.0,
        "masks_dir": str(root / "masks"), "gt_dir": str(gt_dir),
        "agnostic": False,
        "arms": {"A0_anchor": True, "A1_roi": False,
                 "A2_preedit_fb": True, "A2c_preedit_blurall": True,
                 "A4_preedit_fb_roi": False},
        "sequences": seq_cfg,
        "detector_weights": "frcnn",  # Faster R-CNN (MPEG VCM CTC anchor family)
        "detector_classes": [1],  # torchvision native COCO ids: person = 1
    }
    cfg_path = root / "e1_real.yaml"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg, cfg_path


def id2img_by_file(sid: str, fname: str) -> int:
    """train.json file_name is '<sid>/img1/<fname>.jpg' — resolved via cache."""
    global _FILE2ID
    return _FILE2ID[f"{sid}/img1/{fname}"]


_FILE2ID: dict = {}


def main(shard: int = 0, frame_limit: int = 150, qp_grid: list | None = None) -> None:
    """Shard mode (user request): 1 account = 1 sequence x full grid.
    21 shards -> 7 sequences x 3 replicates (replicate = shard // 7)."""
    seq = SEQS[shard % len(SEQS)]
    replicate = shard // len(SEQS)
    qps = qp_grid or QPS
    summary["shard"] = {"index": shard, "seq": seq, "replicate": replicate,
                        "frame_limit": frame_limit, "qp_grid": qps}
    t0 = time.time()
    (WORK / "logs").mkdir(exist_ok=True)
    if not REPO.exists():
        run("clone", ["git", "clone", "https://github.com/wagur1/scope.git", str(REPO)],
            cwd=WORK)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    summary["git_sha"] = sha
    record_gpu(summary)
    run("pip", ["python", "-m", "pip", "install", "-q", "-r", "requirements.txt",
                "ultralytics"], cwd=REPO)
    import sys
    if "/kaggle/working/scope" not in sys.path:
        sys.path.insert(0, "/kaggle/working/scope")  # prepare() imports scope_vcm
    global DATASET
    DATASET = find_dataset_root()
    summary["dataset_root"] = str(DATASET)

    global _FILE2ID
    train = json.loads((DATASET / "annotations" / "train.json").read_text())
    _FILE2ID = {im["file_name"]: im["id"] for im in train["images"]}

    cfg, cfg_path = prepare(seq, frame_limit, qps)
    run("run_e1", ["python", "scripts/run_e1.py", cfg_path], cwd=REPO, check=False)
    run("eval_e1", ["python", "scripts/eval_e1.py", cfg_path,
                    "--weights", "frcnn"], cwd=REPO, check=False)
    rep = run("report_e1", ["python", "scripts/report_e1.py",
                            str(Path(cfg["workdir"])), "--config", cfg_path],
              cwd=REPO, check=False)
    (WORK / "report.md").write_text(rep)
    summary["verdict"] = next((l for l in rep.splitlines() if "VERDICT" in l),
                              "verdict line not found")

    out = WORK / "out"
    out.mkdir(exist_ok=True)
    res = Path(cfg["workdir"]) / "results"
    for f in res.glob("*.csv"):
        shutil.copy2(f, out / f.name)
    summary["result_rows"] = {f.name: sum(1 for _ in f.open()) for f in res.glob("*.csv")}
    enc = Path(cfg["workdir"]) / "enc"
    if enc.exists():
        shutil.rmtree(enc, ignore_errors=True)
    for d in ("scope", "real_e1"):
        shutil.rmtree(WORK / d, ignore_errors=True)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--frame-limit", type=int, default=150)
    ap.add_argument("--qp-grid", default="")
    a = ap.parse_args()
    qg = [int(x) for x in a.qp_grid.split(",") if x] or None
    main(shard=a.shard, frame_limit=a.frame_limit, qp_grid=qg)
