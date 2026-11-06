"""Encoder wrappers for E1 arms.

Design rules (from the code-review pass, 2026-11-07):
  * All x265 arms run fixed-QP (CQP) with AQ and cutree DISABLED, so that a
    nominal ROI offset is actually applied (an ROI map under AQ is not the
    MATH.md section 3 baseline). Same param set for A0/A1/A2/A4.
  * The ROI baseline must FAIL LOUDLY: if the encoder does not accept the
    per-block QP map, raise — A1 must never silently degenerate into A0
    (council R3 fatal-1 protection). `scripts/check_encoders.py` includes a
    bitrate-ordering self-test that pins the map's sign empirically.
  * Bitrate is computed from a caller-supplied frame count — never from a
    sibling-path guess.

VERIFY markers: flags whose spelling/format differs across versions are
marked; `check_encoders.py` is the gate that turns VERIFY into verified.
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CTU_SIZE = 64  # x265 LCU and VTM CTU size for the configs we target


@dataclass
class EncodeResult:
    bitrate_kbps: float
    bitstream: Path
    decoded_dir: Path
    n_frames: int


@dataclass
class EncoderCfg:
    codec: str = "x265"          # x264 | x265 | vtm | vvenc
    qp: int = 28
    preset: str = "medium"       # VERIFY: preset affects output; pinned in manifest
    roi_delta: int = 0           # background QP offset magnitude (MATH.md section 3); 0 = off
    roi_map: Path | None = None  # per-LCU map file (see write_qp_delta_map)
    lossless: bool = False       # negligible-compression ceiling encode (section 4)
    extra_params: dict = field(default_factory=dict)


def _run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    lowered = (proc.stdout + proc.stderr).lower()
    if "unknown option" in lowered or "unknown param" in lowered:
        raise RuntimeError(
            f"encoder rejected an option (see {log}) — refusing to produce "
            "silently-degraded results. VERIFY the flag set for this encoder version."
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}; see {log}")


def write_qp_delta_map(masks: list["np.ndarray"], delta: int, out_path: Path,
                       frame_size: tuple[int, int], tau: float = 0.5) -> Path:
    """Per-LCU background QP offsets for a whole sequence (MATH.md section 3).

    Candidate format (VERIFY against the target encoder before trusting):
    one line per frame; each line holds one integer per LCU in raster order.
    Background LCU (mask mean < tau) gets +delta (higher QP = fewer bits),
    object LCU gets 0. The SIGN and granularity are pinned empirically by the
    check_encoders.py self-test, not by this file's convention.
    """
    h, w = frame_size
    rows, cols = -(-h // CTU_SIZE), -(-w // CTU_SIZE)
    lines = []
    for mask in masks:
        vals = []
        for r in range(rows):
            for c in range(cols):
                block = mask[r * CTU_SIZE:(r + 1) * CTU_SIZE,
                             c * CTU_SIZE:(c + 1) * CTU_SIZE]
                vals.append(str(delta) if block.size and block.mean() < tau else "0")
        lines.append(" ".join(vals))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _x265_params(cfg: EncoderCfg) -> str:
    """Fixed-QP param string shared by every x265 arm (pre-registered policy)."""
    if cfg.lossless:
        return "lossless=1"
    # VERIFY: aq-mode=0/cutree=0 exist across x265 >= 3.x; pinned here so the
    # ROI map applies nominal offsets (council/code-review requirement).
    params = ["qp=" + str(cfg.qp), "aq-mode=0", "cutree=0"]
    if cfg.roi_map is not None and cfg.roi_delta:
        # VERIFY: qp-delta-map is NOT in public x265 (verified against
        # x265cli.h master + Release_3.5 and ffmpeg libx265.c). This path only
        # activates on toolchains where the option exists (VCM/vendor builds);
        # stock toolchains fail loudly below instead of degrading A1 to A0.
        rel = cfg.roi_map.resolve().as_posix().replace(":", r"\:")  # escape drive colon
        params.append(f"qp-delta-map={rel}")
    params += [f"{k}={v}" for k, v in cfg.extra_params.items()]
    return ":".join(params)


def encode(cfg: EncoderCfg, frames_dir: Path, out_dir: Path, n_frames: int,
           fps: float = 30.0) -> EncodeResult:
    """Encode a %04d.png frame directory. Raises on toolchain gaps — no silent fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH — run scripts/check_encoders.py")
    n_frames = int(n_frames)
    if n_frames < 2:
        raise ValueError(f"need >=2 frames, got {n_frames}")

    ext = {"x264": "h264", "x265": "hevc"}.get(cfg.codec)
    if ext is None:
        raise NotImplementedError(
            "VTM/VVenC wrappers are E2 work: build the encoders, then wire the "
            "CTC low-delay cfg + per-CTU ROI path here (PROTOCOL.md E2 step 1)."
        )
    bitstream = out_dir / f"enc.{ext}"
    decoded = out_dir / "decoded"
    decoded.mkdir(parents=True, exist_ok=True)
    src = (frames_dir / "%04d.png").as_posix()

    if cfg.codec == "x264":
        qargs = ["-qp", "0"] if cfg.lossless else ["-qp", str(cfg.qp)]
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", src, "-c:v", "libx264",
               *qargs, "-preset", cfg.preset, "-pix_fmt",
               "yuv444p" if cfg.lossless else "yuv420p", bitstream.as_posix()]
    else:  # x265
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", src, "-c:v", "libx265",
               "-x265-params", _x265_params(cfg), "-preset", cfg.preset,
               "-pix_fmt", "yuv444p" if cfg.lossless else "yuv420p",
               bitstream.as_posix()]
        if cfg.roi_map is not None and cfg.roi_delta:
            logger.warning("x265 ROI map requested — VERIFY support on this "
                           "toolchain (see module docstring); failure here is intentional.")
    _run(cmd, out_dir / "encode.log")

    _run([ffmpeg, "-y", "-i", bitstream.as_posix(), (decoded / "%04d.png").as_posix()],
         out_dir / "decode.log")
    n_decoded = len(list(decoded.glob("*.png")))
    if n_decoded != n_frames:
        raise RuntimeError(f"decode produced {n_decoded} frames, expected {n_frames}")

    size_bits = bitstream.stat().st_size * 8
    bitrate_kbps = size_bits / (n_frames / fps) / 1000.0
    if not math.isfinite(bitrate_kbps) or bitrate_kbps <= 0:
        raise RuntimeError(f"non-positive bitrate for {bitstream}")
    return EncodeResult(bitrate_kbps, bitstream, decoded, n_frames)
