#!/usr/bin/env python
"""Smoke-test the installed toolchain before trusting any E1 numbers.

Checks, in order:
  1. ffmpeg present, libx264/libx265 lossless encodes round-trip (decode +
     pixel-identical check on yuv444p)
  2. fixed-QP x265 param set accepted (qp=, aq-mode=0, cutree=0)
  3. ROI delta-QP map: encode with an all-background map at delta=20 vs A0 at
     the same QP — bitrate MUST drop measurably, which pins BOTH the map's
     acceptance AND its sign empirically. On stock x265 (no qp-delta-map
     support, verified against x265cli.h + ffmpeg libx265.c) this check FAILS
     loudly with instructions — that is the intended behavior: arm A1 must
     never silently degenerate into A0 (council R3 fatal-1 protection).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.codec import EncoderCfg, encode, write_qp_delta_map  # noqa: E402


def _frames(d: Path, n: int = 6) -> tuple[Path, list[np.ndarray]]:
    frames = d / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    imgs = []
    for t in range(n):
        img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(frames / f"{t:04d}.png")
        imgs.append(img.astype(np.float32) / 255.0)
    return frames, imgs


def _lossless_roundtrip(ffmpeg: str, codec_args: list[str], frames: Path, d: Path) -> bool:
    out = d / "ll.mkv"
    r = subprocess.run([ffmpeg, "-y", "-framerate", "30", "-i", str(frames / "%04d.png"),
                        "-c:v", *codec_args, "-pix_fmt", "yuv444p", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False
    dec = d / "dec"
    dec.mkdir()
    subprocess.run([ffmpeg, "-y", "-i", str(out), str(dec / "%04d.png")],
                   capture_output=True, text=True)
    try:
        for t in range(6):
            a = np.asarray(Image.open(frames / f"{t:04d}.png"), dtype=np.int16)
            b = np.asarray(Image.open(dec / f"{t:04d}.png"), dtype=np.int16)
            if not np.array_equal(a, b):
                return False
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    ok = True
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg or 'MISSING'}")
    if not ffmpeg:
        return 1

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        frames, _ = _frames(d)

        for name, args in [("libx264", ["libx264", "-qp", "0"]),
                           ("libx265", ["libx265", "-x265-params", "lossless=1"])]:
            rt = _lossless_roundtrip(ffmpeg, args, frames, d)
            ok &= rt
            print(f"{name} lossless roundtrip (yuv444p, pixel-exact): {'OK' if rt else 'FAIL'}")

        # fixed-QP param set
        r = subprocess.run([ffmpeg, "-y", "-framerate", "30", "-i", str(frames / "%04d.png"),
                            "-c:v", "libx265", "-x265-params", "qp=28:aq-mode=0:cutree=0",
                            "-pix_fmt", "yuv420p", "-f", "null", "-"],
                           capture_output=True, text=True)
        qp_ok = r.returncode == 0 and "unknown option" not in (r.stdout + r.stderr).lower()
        print(f"x265 fixed-QP params (qp/aq-mode=0/cutree=0): {'OK' if qp_ok else 'FAIL'}")
        ok &= qp_ok

        # ROI map self-test: all-background map at delta=20 must reduce bitrate vs A0
        masks = [np.zeros((128, 128), dtype=np.float32) for _ in range(6)]  # all background
        qmap = write_qp_delta_map(masks, 20, d / "qpmap_allbg.txt", (128, 128))
        a0 = encode(EncoderCfg(codec="x265", qp=28, preset="faster"), frames, d / "a0", 6)
        try:
            a1 = encode(EncoderCfg(codec="x265", qp=28, preset="faster", roi_delta=20,
                                   roi_map=qmap), frames, d / "a1", 6)
            if a1.bitrate_kbps < a0.bitrate_kbps * 0.95:
                print(f"ROI map applied (bitrate {a0.bitrate_kbps:.1f} -> {a1.bitrate_kbps:.1f} kbps, "
                      "positive delta reduces rate): OK")
            else:
                print(f"ROI map ACCEPTED but bitrate did not drop "
                      f"({a0.bitrate_kbps:.1f} -> {a1.bitrate_kbps:.1f} kbps) — "
                      "sign or granularity wrong; investigate before E1")
                ok = False
        except RuntimeError as exc:
            print(f"ROI map NOT SUPPORTED on this toolchain ({exc}).\n"
                  "  -> Arm A1 (the thesis baseline) cannot run here yet.\n"
                  "  -> Options: x265 build with qp-delta-map (VCM/vendor), VTM with a "
                  "verified per-CTU QP path, or VVenC ROI (VERIFY each). E1 is BLOCKED "
                  "until one of these passes this check — by design.")
            ok = False

    print("\nPASS" if ok else "\nFAIL — resolve above before running E1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
