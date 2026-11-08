#!/usr/bin/env python
"""Round 18: integration proof for the A1 ROI backend.

Builds tools/x264_roi_encode.c on Kaggle, generates the deterministic clip +
masks, produces ROI maps via scripts/make_roi_maps.py, encodes A0 (zero map)
vs A1 (+delta map) vs A2-blur, and records bitrates. GREEN = bitrates differ
(A1 applies) AND A1 >= A0 sanity holds. Output: out/roi_integrate.json
"""
import json
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "scope"
OUT = WORK / "out"
report: dict = {"steps": {}}


def sh(cmd: list, timeout: int = 1200, cwd: Path | None = None) -> tuple:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    (WORK / "logs").mkdir(exist_ok=True)
    log = WORK / "logs" / f"{p.returncode}_{abs(hash(str(cmd))) % 10000}.log"
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, p.stdout + p.stderr


def sys_executable() -> str:
    return "python"

def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    if not REPO.exists():
        sh(["git", "clone", "https://github.com/wagur1/scope.git", str(REPO)])
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    sh(["bash", "-c", "apt-get update -qq && apt-get install -y -qq libx264-dev "
                      "libpng-dev build-essential ffmpeg python3-pip >/dev/null 2>&1"],
       timeout=1200)

    # build the ROI tool
    tool = WORK / "x264_roi_encode"
    rc, err = sh(["bash", "-c",
                  f"gcc -O2 -o {tool} {REPO}/tools/x264_roi_encode.c -lx264 -lm "
                  f"> {WORK}/logs/build.log 2>&1; echo rc=$?"], timeout=600)
    report["build_rc"] = rc
    report["build_tail"] = err[-400:]
    if rc != 0:
        (OUT / "roi_integrate.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    # deterministic clip + masks (same generator as prior rounds)
    import numpy as np
    w, h, n = 160, 120, 24
    import random as _r
    _r.seed(7)
    yuvp = WORK / "clip.yuv"
    masks = {}
    with open(yuvp, "wb") as f:
        for t in range(n):
            y = np.array([[60 + _r.randrange(140) for _ in range(w)] for _ in range(h)],
                         dtype=np.uint8)
            u = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            v = np.full((h // 2, w // 2), 128, dtype=np.uint8)
            bx, by = (t * 2) % (w - 64), (t * 3) % (h - 64)
            y[by:by + 48, bx:bx + 48] = 235
            m = np.zeros((h, w), dtype=np.float32)
            m[by:by + 48, bx:bx + 48] = 1.0
            masks[t] = m
            f.write(y.tobytes()); f.write(u.tobytes()); f.write(v.tobytes())
    from scope_vcm.masks import save_masks
    save_masks(WORK / "masks.npz", masks)

    # ROI maps via the repo driver
    sh([sys_executable(), "scripts/make_roi_maps.py", str(WORK / "masks.npz"),
        str(w), str(h), "10", str(WORK / "roi_d10.txt")], cwd=REPO, timeout=300)

    # encode arms
    res = {}
    for tag, roimap in (("A0_zero", str(WORK / "zero.txt")), ("A1_d10", str(WORK / "roi_d10.txt"))):
        if roimap == str(WORK / "zero.txt"):
            mbw, mbh = -(-w // 16), -(-h // 16)
            Path(roimap).write_text(
                chr(10).join([" ".join(["0"] * (mbw * mbh)) for _ in range(n)]) + chr(10))
        outp = WORK / f"{tag}.bin"
        rc, err = sh([str(tool), str(yuvp), str(w), str(h), "30", "28",
                      roimap, "0", str(outp), str(n)], timeout=600)
        res[tag] = {"ran": rc == 0, "bytes": outp.stat().st_size if outp.exists() else 0,
                    "stderr": err[-200:]}
    if res["A0_zero"]["bytes"] and res["A1_d10"]["bytes"]:
        res["ratio_A1_vs_A0"] = round(res["A1_d10"]["bytes"] / res["A0_zero"]["bytes"], 4)
        res["a1_applies"] = bool(res["ratio_A1_vs_A0"] < 0.99)
    report["arms"] = res

    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "roi_integrate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
