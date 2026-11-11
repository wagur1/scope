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
    codec: str = "x265"          # x264 | x264roi | x265 | vtm | vvenc
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


def subprocess_run(cmd: list, timeout: int = 3600) -> tuple:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def _pack_i420(frames_dir: Path, yuvp: Path, n_frames: int,
               frame_size: tuple[int, int]) -> None:
    """PNG frames -> raw I420 (BT.601), matching scripts/pack_yuv.py."""
    from PIL import Image
    files = sorted(frames_dir.glob("*.png"))
    if len(files) != n_frames:
        raise RuntimeError(f"{frames_dir}: {len(files)} pngs != {n_frames}")
    with open(yuvp, "wb") as f:
        for fp in files:
            with Image.open(fp) as image:
                if image.size != (frame_size[1], frame_size[0]):
                    raise ValueError(f"{fp}: frame size {image.size} != "
                                     f"{(frame_size[1], frame_size[0])}")
                im = np.asarray(image.convert("RGB"), dtype=np.float32)
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            y = 0.257 * r + 0.504 * g + 0.098 * b + 16
            u = -0.148 * r - 0.291 * g + 0.439 * b + 128
            v = 0.439 * r - 0.368 * g - 0.071 * b + 128
            f.write(np.clip(y, 0, 255).astype(np.uint8).tobytes())
            f.write(np.clip(u[::2, ::2], 0, 255).astype(np.uint8).tobytes())
            f.write(np.clip(v[::2, ::2], 0, 255).astype(np.uint8).tobytes())


def _zero_map(outp: Path, n_frames: int, frame_size: tuple[int, int]) -> Path:
    h, w = frame_size
    mbw, mbh = (w + 15) // 16, (h + 15) // 16
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for _ in range(n_frames):
            f.write(" ".join(["0"] * (mbw * mbh)) + "\n")
    return outp


def _validate_roi_map(path: Path, n_frames: int,
                      frame_size: tuple[int, int]) -> None:
    h, w = frame_size
    expected = ((w + 15) // 16) * ((h + 15) // 16)
    with path.open() as f:
        for t in range(n_frames):
            values = f.readline().split()
            if len(values) != expected:
                raise ValueError(f"{path}: frame {t} has {len(values)} ROI values, "
                                 f"expected {expected}")
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"{path}: frame {t} has non-finite ROI offsets")
        if f.read().strip():
            raise ValueError(f"{path}: surplus ROI map data after {n_frames} frames")


def write_qp_delta_map(masks, delta: int, out_path: Path,
                       frame_size: tuple[int, int], tau: float = 0.5,
                       n_frames: int | None = None,
                       block_size: int = CTU_SIZE) -> Path:
    """Per-block background QP offsets for a whole sequence (MATH.md section 3).

    One line per frame; each line holds one offset per block in raster order.
    x264roi uses 16x16 macroblocks; the CTU backends retain 64x64 blocks.
    Background blocks (mask mean < tau) get +delta (higher QP = fewer bits),
    object blocks get 0. Backend intervention still needs an empirical gate.

    Streaming: `masks` may be a LazyMasks archive or any indexable — frames
    are pulled one at a time, never materialized as a list (a 1080p x 6000
    frame float32 list is ~50 GB; self-review fix 2).
    """
    h, w = frame_size
    if h <= 0 or w <= 0 or block_size <= 0:
        raise ValueError("frame dimensions and block_size must be positive")
    rows, cols = -(-h // block_size), -(-w // block_size)
    if n_frames is None:
        n_frames = len(masks)  # type: ignore[arg-type]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in range(n_frames):
            mask = masks[t]
            if mask.shape != (h, w):
                raise ValueError(f"mask {t}: shape {mask.shape} != {(h, w)}")
            if not np.isfinite(mask).all():
                raise ValueError(f"mask {t}: non-finite values")
            vals = []
            for r in range(rows):
                for c in range(cols):
                    block = mask[r * block_size:(r + 1) * block_size,
                                 c * block_size:(c + 1) * block_size]
                    vals.append(str(delta) if block.size and block.mean() < tau else "0")
            f.write(" ".join(vals) + "\n")
    return out_path


def _check_decoded(decoded: Path, n_frames: int,
                   frame_size: tuple[int, int] | None) -> None:
    from PIL import Image
    files = sorted(decoded.glob("*.png"))
    expected = [f"{t:04d}.png" for t in range(n_frames)]
    if [f.name for f in files] != expected:
        raise RuntimeError(f"{decoded}: expected {n_frames} zero-indexed decoded frames, "
                           f"found {len(files)}")
    if frame_size is not None:
        for fp in files:
            with Image.open(fp) as image:
                if image.size != (frame_size[1], frame_size[0]):
                    raise RuntimeError(f"{fp}: decoded size {image.size} != "
                                       f"{(frame_size[1], frame_size[0])}")


def _x265_params(cfg: EncoderCfg) -> str:
    """Fixed-QP param string shared by every x265 arm (pre-registered policy)."""
    if cfg.lossless:
        return "lossless=1"
    # VERIFY: aq-mode=0/cutree=0 exist across x265 >= 3.x; pinned here so the
    # ROI map applies nominal offsets (council/code-review requirement).
    # low-delay pinning (review P2-03): bframes=0 + no scenecut — without this
    # the wrapper silently allows reordered B-frames, contradicting the
    # registered low-delay protocol
    params = ["qp=" + str(cfg.qp), "bframes=0", "scenecut=0", "aq-mode=0", "cutree=0"]
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
           fps: float = 30.0, frame_size: tuple[int, int] | None = None) -> EncodeResult:
    """Encode a %04d.png directory; frame_size is (height, width). No fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH — run scripts/check_encoders.py")
    n_frames = int(n_frames)
    if n_frames < 2:
        raise ValueError(f"need >=2 frames, got {n_frames}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")

    if cfg.codec == "x264roi":
        # A1 backend (review P1-04 closed): the verified C tool encodes EVERY
        # arm - zero map = anchor, delta map = ROI intervention - so all arms
        # share one encode path and differ only in the map.
        tool = Path(cfg.extra_params.get("tool_path", ""))
        if not tool.is_file():
            raise FileNotFoundError(f"ROI tool missing: {tool} - build per docs")
        if cfg.lossless:
            raise NotImplementedError("lossless ceiling uses the ffmpeg x264 path")
        if frame_size is None:
            raise ValueError("x264roi requires frame_size=(height, width)")
        h, w = frame_size
        if any(not isinstance(v, (int, np.integer)) or v <= 0 or v % 2
               for v in (h, w)):
            raise ValueError(f"I420 requires positive even dimensions, got {frame_size}")
        if cfg.roi_delta and cfg.roi_map is None:
            raise ValueError("nonzero ROI delta requires a map")
        roimap = (cfg.roi_map if cfg.roi_map is not None else
                  _zero_map(out_dir / "roi_zero.txt", n_frames, frame_size))
        _validate_roi_map(roimap, n_frames, frame_size)
        yuvp = out_dir / "input.yuv"
        _pack_i420(frames_dir, yuvp, n_frames, frame_size)
        binp = out_dir / "enc.h264"
        _run([str(tool), str(yuvp), str(w), str(h), str(fps), str(cfg.qp),
              str(roimap), "0", str(binp), str(n_frames)], out_dir / "encode.log")
        if not binp.is_file() or binp.stat().st_size == 0:
            raise RuntimeError(f"ROI tool produced no bitstream; see {out_dir / 'encode.log'}")
        decoded = out_dir / "decoded"
        decoded.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg, "-y", "-i", binp.as_posix(), "-start_number", "0",
              (decoded / "%04d.png").as_posix()],
             out_dir / "decode.log")
        _check_decoded(decoded, n_frames, frame_size)
        kbps = binp.stat().st_size * 8 / (n_frames / fps) / 1000.0
        return EncodeResult(kbps, binp, decoded, n_frames)

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
        if cfg.roi_map is not None or cfg.roi_delta:
            # External review P1-04: the x264 branch had no ROI wiring — A1/A4 on
            # x264 silently encoded WITHOUT the intervention. Loud reject until a
            # verified x264 ROI backend (C-API quant_offsets) is integrated.
            raise NotImplementedError(
                "x264 ROI delta-QP backend is not integrated (review P1-04): "
                "A1/A4 on x264 would silently encode without the intervention. "
                "Use a verified ROI backend or disable the ROI arms."
            )
        qargs = ["-qp", "0"] if cfg.lossless else ["-qp", str(cfg.qp)]
        # low-delay pinning (review P2-03): bframes=0
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", src, "-c:v", "libx264",
               *qargs, "-preset", cfg.preset, "-bf", "0", "-pix_fmt",
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

    _run([ffmpeg, "-y", "-i", bitstream.as_posix(), "-start_number", "0",
          (decoded / "%04d.png").as_posix()],
         out_dir / "decode.log")
    _check_decoded(decoded, n_frames, frame_size)

    size_bits = bitstream.stat().st_size * 8
    bitrate_kbps = size_bits / (n_frames / fps) / 1000.0
    if not math.isfinite(bitrate_kbps) or bitrate_kbps <= 0:
        raise RuntimeError(f"non-positive bitrate for {bitstream}")
    return EncodeResult(bitrate_kbps, bitstream, decoded, n_frames)
