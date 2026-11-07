#!/usr/bin/env python
"""ROI delta-QP hunt — Bottleneck 1 decided empirically, all in Python
(no bash heredoc quoting; round-1 bash version died on paths + heredocs).

Steps: apt install dev headers -> grep header probes -> gcc compile the two
sign-test programs (kaggle/roi_hunt_x264.c, kaggle/roi_hunt_x265.c from the
repo) -> encode identical clips with/without offsets -> bitrate ratio pins
availability AND sign. Output: out/roi_hunt.json
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "scope"
OUT = WORK / "out"
report: dict = {"candidates": {}, "gcc_errors": {}}


def sh(cmd: list, timeout: int = 900) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)

    if not REPO.exists():
        sh(["git", "clone", "https://github.com/wagur1/scope.git", str(REPO)])

    rc, out = sh(["apt-get", "update", "-qq"])
    rc, out = sh(["apt-get", "install", "-y", "-qq", "libx264-dev", "libx265-dev",
                  "build-essential"], timeout=1200)
    report["apt_rc"] = rc

    h264_h = next(Path("/usr/include").rglob("x264.h"), None)
    h265_h = next(Path("/usr/include").rglob("x265.h"), None)
    report["headers"] = {"x264": str(h264_h), "x265": str(h265_h)}
    if h264_h:
        report["x264_header_quant_offsets"] = h264_h.read_text().count("quant_offsets")
    if h265_h:
        txt = h265_h.read_text()
        report["x265_header_quantOffsets"] = txt.count("quantOffsets")
        report["x265_header_qp_delta_map"] = txt.count("qp-delta-map") + txt.count("qpDeltaMap")

    def sign_test(lib: str, use_off: int, tag: str) -> dict:
        exe = WORK / f"hunt_{lib}_{tag}"
        src = REPO / "kaggle" / f"roi_hunt_{lib}.c"
        rc, err = sh(["gcc", "-O2", "-o", str(exe), str(src), f"-l{lib}", "-lm"])
        if rc != 0:
            report["gcc_errors"][f"{lib}_{tag}"] = err[-2000:]
            return {"compiled": False}
        binp = WORK / f"{lib}_{tag}.bin"
        rc, err = sh([str(exe), "160", "120", "24", "28", str(use_off), str(binp)],
                     timeout=600)
        if rc != 0:
            report["gcc_errors"][f"{lib}_{tag}_run"] = err[-2000:]
            return {"compiled": True, "ran": False, "stderr": err[-800:]}
        return {"compiled": True, "ran": True, "bytes": binp.stat().st_size}

    for lib, field_probe in (("x264", "x264_header_quant_offsets"),
                             ("x265", "x265_header_quantOffsets")):
        if report.get(field_probe, 0) > 0:
            base = sign_test(lib, 0, "base")
            off = sign_test(lib, 1, "off")
            entry = {"base": base, "offset": off}
            if base.get("ran") and off.get("ran"):
                ratio = off["bytes"] / base["bytes"]
                entry["ratio"] = round(ratio, 4)
                entry["works_and_sign_positive"] = bool(ratio < 0.95)
            report["candidates"][lib] = entry
        else:
            report["candidates"][lib] = {"declared_in_header": False}

    try:
        import av  # noqa: F401
        report["pyav"] = av.__version__
    except ImportError:
        report["pyav"] = None

    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "roi_hunt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
