"""E1 runner: sequence x arm x QP grid -> results/e1.csv (resumable).

Streaming policy (code-review fix): frames are processed ONE AT A TIME —
never a whole-sequence np.stack (MOT17-02 is ~6000 frames x 1080p).
Masks are accessed lazily per frame index from the cached npz.

Pre-registered pins (docs/MATH.md sections 2-6):
  * one mask file per sequence, shared by every arm
  * blur sigma fixed in the docs, not tuned post-hoc
  * manifest is appended per pass (timestamped), never overwritten
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .codec import EncoderCfg, encode, write_qp_delta_map
from .masks import LazyMasks
from .preedit import build_editor

logger = logging.getLogger(__name__)

FIELDS = ["seq", "arm", "qp", "delta", "rate_kbps", "quality", "is_ceiling", "notes"]


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _save_frame(img, out: Path, t: int) -> None:
    from PIL import Image
    import numpy as np
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(out / f"{t:04d}.png")


def _load_frame(frames_dir: Path, t: int):
    from PIL import Image
    import numpy as np
    f = frames_dir / f"{t:04d}.png"
    img = Image.open(f)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{f}: expected HxWx3 RGB, got {arr.shape}")
    return arr


def _frame_count(frames_dir: Path) -> int:
    n = len(list(frames_dir.glob("*.png")))
    if n < 2:
        raise ValueError(f"{frames_dir}: need >=2 frames, found {n}")
    return n


def run_e1(config_path: str | Path) -> Path:
    import datetime

    cfg = yaml.safe_load(Path(config_path).read_text())
    root = Path(cfg["workdir"]).resolve()
    results_csv = root / "results" / "e1.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if results_csv.exists():
        with results_csv.open() as f:
            for row in csv.DictReader(f):
                done.add((row["seq"], row["arm"], int(row["qp"]), int(row["delta"])))
        logger.info("resuming: %d rows already present", len(done))

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    manifest_dir = root / "results" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"run_{stamp}.json").write_text(json.dumps({
        "git_sha": _git_sha(Path(__file__).parent.parent),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "started": stamp,
        "encoder": cfg.get("encoder"),
        "preset": cfg.get("preset"),
        "qp_grid": cfg.get("qp_grid"),
        "roi_deltas": cfg.get("roi_deltas"),
        "blur_sigma": cfg.get("blur_sigma"),
    }, indent=2))

    qp_grid = list(cfg["qp_grid"])
    deltas = [0] + [int(d) for d in cfg.get("roi_deltas", [4, 8, 12, 16]) if int(d) > 0]
    enc_codec = cfg.get("encoder", "x265")
    if enc_codec == "x264roi" and not cfg.get("tool_path"):
        raise ValueError("encoder x264roi requires tool_path (compiled x264_roi_encode)")
    preset = cfg.get("preset", "medium")
    fps = float(cfg.get("fps", 30.0))  # bitrate depends on it — pin in config, not code
    arms = cfg.get("arms", {})
    blur_sigma = float(cfg.get("blur_sigma", 8.0))  # pre-registered in MATH.md section 2/m6

    new_rows = 0
    with results_csv.open("a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if fcsv.tell() == 0:
            writer.writeheader()

        for seq in cfg["sequences"]:
            sid = seq["id"]
            frames_dir = Path(seq["frames"]).resolve()
            if not frames_dir.exists():
                raise FileNotFoundError(
                    f"{sid}: frames dir {frames_dir} missing — fill data paths in the config"
                )
            n_frames = _frame_count(frames_dir)
            mask_root = Path(cfg.get("masks_dir", root / "masks"))
            masks = LazyMasks(mask_root / f"{sid}.npz", n_frames)
            h, w = _load_frame(frames_dir, 0).shape[:2]
            frames_rgba_note = ""

            def _emit(arm: str, qp: int, delta: int, src_dir: Path,
                      roi_map: Path | None = None) -> None:
                nonlocal new_rows
                key = (sid, arm, int(qp), int(delta))
                if key in done:
                    return
                out = root / "enc" / arm / sid / f"qp{qp}_d{delta}"
                ecfg = EncoderCfg(codec=enc_codec, qp=qp, preset=preset,
                                  roi_delta=delta if roi_map is not None else 0,
                                  roi_map=roi_map,
                                  extra_params={"tool_path": cfg.get("tool_path", "")})
                res = encode(ecfg, src_dir, out, n_frames=n_frames, fps=fps)
                writer.writerow({"seq": sid, "arm": arm, "qp": qp, "delta": delta,
                                 "rate_kbps": round(res.bitrate_kbps, 3),
                                 "quality": "", "is_ceiling": 0, "notes": ""})
                fcsv.flush()
                done.add(key)
                new_rows += 1

            def _roi_map_for(arm: str, qp: int, d: int) -> Path:
                # LazyMasks streams per-frame access — no list materialization
                return write_qp_delta_map(
                    masks, d,
                    root / "enc" / arm / sid / "maps" / f"qp{qp}_d{d}.txt",
                    (h, w), n_frames=n_frames)

            # A0 anchor + A1 roi: original frames (A1 gets per-LCU maps from shared masks)
            if arms.get("A0_anchor", True):
                for qp in qp_grid:
                    _emit("A0_anchor", qp, 0, frames_dir)
            if arms.get("A1_roi", True):
                for qp in qp_grid:
                    for d in deltas[1:]:
                        _emit("A1_roi", qp, d, frames_dir, roi_map=_roi_map_for("A1_roi", qp, d))

            # Pre-edit arms: stream frame-by-frame (no whole-sequence stack)
            def _preedit_to_dir(editor, arm: str) -> Path:
                edited_dir = root / "edited" / arm / sid
                if edited_dir.exists() and len(list(edited_dir.glob("*.png"))) == n_frames:
                    return edited_dir
                edited_dir.mkdir(parents=True, exist_ok=True)
                editor.begin_sequence(sid)
                for t in range(n_frames):
                    img = _load_frame(frames_dir, t)
                    _save_frame(editor.edit_frame(img, t), edited_dir, t)
                return edited_dir

            if arms.get("A2_preedit_fb", True):
                editor = build_editor({"type": "blurbg", "sigma": blur_sigma},
                                      mask_provider=lambda s, t, _m=masks: _m[t])
                edited_dir = _preedit_to_dir(editor, "A2_preedit_fb")
                for qp in qp_grid:
                    _emit("A2_preedit_fb", qp, 0, edited_dir)

            if arms.get("A2c_preedit_blurall", False):
                # mechanism control (MATH.md section 2): mask == 0 -> blur everything
                editor = build_editor({"type": "blurbg", "sigma": blur_sigma},
                                      mask_provider=lambda s, t: np.zeros((h, w), dtype=np.float32))
                edited_dir = _preedit_to_dir(editor, "A2c_preedit_blurall")
                for qp in qp_grid:
                    _emit("A2c_preedit_blurall", qp, 0, edited_dir)

            if arms.get("A3_preedit_scope", False):
                editor = build_editor({
                    "type": "scope",
                    "scope_repo": cfg["scope_repo"],
                    "ckpt_path": cfg["scope_ckpt"],
                    "family": cfg.get("scope_family", "diff_jpeg"),
                    "severity": float(cfg.get("scope_severity", 0.5)),
                })
                edited_dir = _preedit_to_dir(editor, "A3_preedit_scope")
                for qp in qp_grid:
                    _emit("A3_preedit_scope", qp, 0, edited_dir)

            if arms.get("A4_preedit_fb_roi", True) and arms.get("A2_preedit_fb", True):
                edited_dir = root / "edited" / "A2_preedit_fb" / sid
                for qp in qp_grid:
                    for d in deltas[1:]:
                        _emit("A4_preedit_fb_roi", qp, d, edited_dir,
                              roi_map=_roi_map_for("A4_preedit_fb_roi", qp, d))

    logger.info("E1 encode pass finished: %d new rows -> %s", new_rows, results_csv)
    return results_csv
