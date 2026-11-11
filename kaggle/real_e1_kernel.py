#!/usr/bin/env python
"""Real-data E1 kernel: MOT17 (FRCNN variants) via the mot17-coco dataset.

Design (one kernel = full 7-sequence pass, ~1.5-2h on Kaggle CPU):
  * frames: first 150 per sequence, resized to 960x540 (encode-budget smoke;
    the full-resolution full-length grid is post-gate work)
  * arms: A0 / A1 / A2 / A2c / A3 on the shared x264 ROI tool; A4 is gated
  * rate control: qp_grid supplies CRF values, AQ mode 1 / strength 1.0,
    mbtree off, x264 defaults (no named preset), single-thread / no B-frames
  * scorer: frozen Faster R-CNN, person class only, vs MOT GT (pedestrian), scaled
    to the resized frames
  * all pool accounts run the IDENTICAL job -> replicates double as a
    cross-machine determinism check on real encoder output

Output: out/{e1.csv, quality.csv, ceiling.csv, report.md, summary.json}
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

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
    log = WORK / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        log.write_text(f"$ {' '.join(map(str, cmd))}\n\n{exc}\n", encoding="utf-8")
        summary["steps"][name] = {"returncode": None, "error": str(exc),
                                  "seconds": round(time.time() - t0, 1)}
        raise
    log.write_text(f"$ {' '.join(map(str, cmd))}\n\n--- stdout ---\n"
                   f"{proc.stdout}\n--- stderr ---\n{proc.stderr}", encoding="utf-8")
    summary["steps"][name] = {"returncode": proc.returncode,
                              "seconds": round(time.time() - t0, 1)}
    print(f"[{name}] rc={proc.returncode} ({summary['steps'][name]['seconds']}s)")
    if check and proc.returncode != 0:
        raise RuntimeError(f"{name} failed — see {log}")
    return proc.stdout + proc.stderr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_scope_assets() -> tuple[Path, Path]:
    """Fetch into a new kernel-owned path; keep code available for A3 inference."""
    asset_root = Path(tempfile.mkdtemp(prefix="scope_assets_", dir=WORK))
    scope_dir = asset_root / "SCOPE"
    # Never reuse/prune an existing user's WORK/SCOPE or its runs directory.
    summary["scope_assets_dir"] = str(asset_root)
    run("scope_clone", ["git", "clone", "--depth", "1",
                        "https://github.com/miccunifi/SCOPE.git", scope_dir],
        cwd=WORK, timeout=600)
    inference = scope_dir / "scope" / "inference.py"
    if not inference.is_file() or inference.stat().st_size == 0:
        raise FileNotFoundError(f"SCOPE inference source missing or empty: {inference}")
    summary["scope_source_sha256"] = sha256(inference)
    summary["scope_git_sha"] = run("scope_sha", ["git", "rev-parse", "HEAD"],
                                   cwd=scope_dir).strip()
    run("scope_pip", ["python", "-m", "pip", "install", "-q", "gdown"], cwd=WORK)
    runs = scope_dir / "runs"
    run("scope_weights", ["python", "-m", "gdown", "--folder",
                          "https://drive.google.com/drive/folders/1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES",
                          "-O", runs], cwd=WORK, timeout=1800)
    checkpoints = sorted(p for p in runs.rglob("best_val.pt")
                         if p.is_file() and p.stat().st_size > 0)
    if not checkpoints:
        raise FileNotFoundError(f"SCOPE requires a nonempty best_val.pt under {runs}")
    selected = checkpoints[0]
    scope_ckpt = asset_root / "scope_ckpt.pt"
    shutil.copy2(selected, scope_ckpt)
    if not scope_ckpt.is_file() or scope_ckpt.stat().st_size == 0:
        raise FileNotFoundError(f"SCOPE checkpoint copy missing or empty: {scope_ckpt}")
    summary["scope_ckpt_origin"] = selected.relative_to(scope_dir).as_posix()
    summary["scope_ckpt"] = str(scope_ckpt)
    summary["scope_ckpt_sha256"] = sha256(scope_ckpt)
    # Only downloaded ablation weights are pruned. Inference source survives
    # until encode/eval/report have all succeeded; failures keep it for diagnosis.
    shutil.rmtree(runs)
    return scope_dir, scope_ckpt


def build_config(root: Path, sequences: list, qps: list,
                 scope_dir: Path, scope_ckpt: Path) -> dict:
    """Require A3 assets, then pin the complete approved arm/ROI grid."""
    tool = WORK / "tools" / "x264_roi_encode"
    for asset in (tool, scope_dir / "scope" / "inference.py", scope_ckpt):
        if not asset.is_file() or asset.stat().st_size == 0:
            raise FileNotFoundError(f"Required experiment asset missing or empty: {asset}")
    return {
        "workdir": str(root / "work"),
        "encoder": "x264roi", "tool_path": str(tool), "fps": 30.0,
        # The C backend ignores EncoderCfg.preset and calls x264_param_default.
        "preset": None,
        "encoder_policy": {"rate_control": "CRF", "grid_parameter": "qp_grid",
                           "aq_mode": 1, "aq_strength": 1.0, "mbtree": False,
                           "parameter_defaults": "x264_param_default",
                           "named_preset": None, "threads": 1, "bframes": 0},
        "qp_grid": qps, "roi_deltas": [4, 8, 12, 16], "blur_sigma": 8.0,
        "masks_dir": str(root / "masks"), "gt_dir": str(root / "gt"),
        "agnostic": False,
        "arms": {"A0_anchor": True, "A1_roi": True,
                 "A2_preedit_fb": True, "A2c_preedit_blurall": True,
                 "A3_preedit_scope": True, "A4_preedit_fb_roi": False},
        "scope_repo": str(scope_dir), "scope_ckpt": str(scope_ckpt),
        "scope_family": "diff_jpeg", "scope_severity": 0.5,
        "sequences": sequences,
        "detector_weights": "frcnn",  # MPEG VCM CTC anchor family
        "detector_classes": [1],  # torchvision COCO ids: person = 1
    }


def prepare(seq_name: str, frame_limit: int, qps: list) -> tuple[dict, Path]:
    """Frames (resized) + per-seq COCO GT (scaled) + YOLO masks + e1 config."""
    import cv2
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

    scope_dir, scope_ckpt = prepare_scope_assets()
    cfg = build_config(root, seq_cfg, qps, scope_dir, scope_ckpt)
    cfg_path = root / "e1_real.yaml"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg, cfg_path


def id2img_by_file(sid: str, fname: str) -> int:
    """train.json file_name is '<sid>/img1/<fname>.jpg' — resolved via cache."""
    global _FILE2ID
    return _FILE2ID[f"{sid}/img1/{fname}"]


_FILE2ID: dict = {}


def publish_success(cfg: dict, report: str, started: float) -> None:
    """Keep the existing small out/ and first-sorting deliverables."""
    out = WORK / "out"
    deliv = WORK / "0000deliv"
    out.mkdir(exist_ok=True)
    deliv.mkdir(exist_ok=True)
    res = Path(cfg["workdir"]) / "results"
    rows = {}
    for path in res.glob("*.csv"):
        shutil.copy2(path, out / path.name)
        with path.open() as stream:
            rows[path.name] = sum(1 for _ in stream)
    summary["result_rows"] = rows
    summary["status"] = "success"
    summary["elapsed_s"] = round(time.time() - started, 1)
    (WORK / "report.md").write_text(report, encoding="utf-8")
    (out / "report.md").write_text(report, encoding="utf-8")
    text = json.dumps(summary, indent=2)
    (out / "summary.json").write_text(text, encoding="utf-8")
    for path in list(out.glob("*.csv")) + [out / "summary.json", out / "report.md"]:
        shutil.copy2(path, deliv / path.name)
    (WORK / "summary.json").write_text(text, encoding="utf-8")
    print(text)


def archive_previous_success() -> Path | None:
    """Move only prior kernel-published files out of the current failure view."""
    archive = None
    for label, dest in (("root", WORK), ("out", WORK / "out"),
                        ("0000deliv", WORK / "0000deliv")):
        previous_summary = dest / "summary.json"
        if not previous_summary.is_file():
            continue
        try:
            previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        # A publication summary identifies kernel-owned historical deliverables.
        # Legacy summaries had result_rows but no status; preserve those too.
        if previous.get("status") != "success" and not (
                "status" not in previous and "result_rows" in previous):
            continue
        names = ["summary.json", "report.md"]
        if label != "root":
            names += [name for name in previous.get("result_rows", {})
                      if Path(name).name == name and Path(name).suffix == ".csv"]
        if archive is None:
            archive = Path(tempfile.mkdtemp(prefix="previous_deliverables_", dir=WORK))
        target = archive / label
        target.mkdir()
        for name in names:
            path = dest / name
            if path.is_file():
                shutil.move(str(path), str(target / name))
    return archive


def publish_failure(exc: Exception, started: float) -> None:
    """Publish isolated diagnostics, preserving historical results and source."""
    previous = archive_previous_success()
    if previous is not None:
        summary["previous_deliverables"] = str(previous)
    summary["status"] = "failed"
    summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
    summary.pop("verdict", None)
    summary.pop("result_rows", None)
    summary["elapsed_s"] = round(time.time() - started, 1)
    log_dir = WORK / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    # Small API-visible log; full subprocess logs remain in logs/ for diagnosis.
    details = ["Experiment FAILED; no successful result is claimed by this run.\n",
               f"{type(exc).__name__}: {exc}\n", "Source, logs and partial artifacts retained.\n"]
    for name, step in summary["steps"].items():
        if step.get("returncode") != 0:
            path = log_dir / f"{name}.log"
            details.append(f"\n--- {name} ---\n")
            if path.is_file():
                details.append(path.read_text(encoding="utf-8", errors="replace")[-4000:])
    text = json.dumps(summary, indent=2)
    for dest in (WORK, WORK / "out", WORK / "0000deliv"):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "summary.json").write_text(text, encoding="utf-8")
        (dest / "failure.log").write_text("".join(details), encoding="utf-8")


def main(shard: int = 0, frame_limit: int = 150, qp_grid: list | None = None) -> None:
    """One account = one sequence x full grid; shard // 7 is the replicate."""
    global DATASET, _FILE2ID
    seq = SEQS[shard % len(SEQS)]
    qps = qp_grid or QPS
    summary.clear()
    summary.update({"steps": {}, "status": "running",
                    "shard": {"index": shard, "seq": seq, "replicate": shard // len(SEQS),
                              "frame_limit": frame_limit, "qp_grid": qps}})
    t0 = time.time()
    repo_owned = not REPO.exists()
    frames_owned = not (WORK / "real_e1").exists()
    try:
        (WORK / "logs").mkdir(parents=True, exist_ok=True)
        if repo_owned:
            run("clone", ["git", "clone", "https://github.com/wagur1/scope.git", REPO],
                cwd=WORK)
        summary["git_sha"] = run("git_sha", ["git", "rev-parse", "HEAD"], cwd=REPO).strip()
        record_gpu(summary)
        (WORK / "tools").mkdir(parents=True, exist_ok=True)
        tool = WORK / "tools" / "x264_roi_encode"
        source = REPO / "tools" / "x264_roi_encode.c"
        summary["tool_source_sha256"] = sha256(source)
        # Separate checked commands: no shell suffix can mask apt/gcc failures.
        run("build_deps", ["apt-get", "install", "-y", "-qq", "libx264-dev"],
            cwd=WORK, timeout=900)
        build_log = run("build_tool", ["gcc", "-O2", "-o", tool, source, "-lx264", "-lm"],
                        cwd=WORK, timeout=900)
        summary["tool_built"] = tool.is_file()
        summary["tool_build_tail"] = build_log[-300:]
        if not tool.is_file():
            raise FileNotFoundError(f"ROI build did not produce a regular file: {tool}")
        summary["tool_path"] = str(tool)
        summary["tool_sha256"] = sha256(tool)
        run("pip", ["python", "-m", "pip", "install", "-q", "-r", "requirements.txt",
                    "ultralytics"], cwd=REPO)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))  # prepare() imports scope_vcm lazily
        DATASET = find_dataset_root()
        summary["dataset_root"] = str(DATASET)
        train = json.loads((DATASET / "annotations" / "train.json").read_text())
        _FILE2ID = {im["file_name"]: im["id"] for im in train["images"]}

        cfg, cfg_path = prepare(seq, frame_limit, qps)
        summary["encoder_policy"] = cfg["encoder_policy"]
        run("run_e1", ["python", "scripts/run_e1.py", cfg_path], cwd=REPO)
        run("eval_e1", ["python", "scripts/eval_e1.py", cfg_path,
                        "--weights", "frcnn"], cwd=REPO)
        report = run("report_e1", ["python", "scripts/report_e1.py",
                                   str(Path(cfg["workdir"])), "--config", cfg_path], cwd=REPO)
        summary["verdict"] = next((line for line in report.splitlines() if "VERDICT" in line),
                                  "verdict line not found")
        publish_success(cfg, report, t0)
        # Only after all stages and publication succeed: retain the small outputs,
        # and remove kernel-owned scratch. Never delete pre-existing user assets.
        if frames_owned:
            shutil.rmtree(WORK / "real_e1", ignore_errors=True)
        if repo_owned:
            shutil.rmtree(REPO, ignore_errors=True)
        if summary.get("scope_assets_dir"):
            shutil.rmtree(Path(summary["scope_assets_dir"]), ignore_errors=True)
    except Exception as exc:
        publish_failure(exc, t0)
        raise


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--frame-limit", type=int, default=150)
    ap.add_argument("--qp-grid", default="")
    a = ap.parse_args()
    qg = [int(x) for x in a.qp_grid.split(",") if x] or None
    main(shard=a.shard, frame_limit=a.frame_limit, qp_grid=qg)
