#!/usr/bin/env python
"""ROI delta-QP hunt — Bottleneck 1 decided empirically, all in Python
(no bash heredoc quoting; round-1 bash version died on paths + heredocs).

Steps: apt install dev headers -> grep header probes -> gcc compile the two
sign-test programs (kaggle/roi_hunt_x264.c, kaggle/roi_hunt_x265.c from the
repo) -> encode identical clips with/without offsets -> bitrate ratio pins
availability AND sign. Output: out/roi_hunt.json
"""
import json
import os
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

    def sign_test(lib: str, offv: float, tag: str) -> dict:
        exe = WORK / f"hunt_{lib}_{tag}"
        src = REPO / "kaggle" / f"roi_hunt_{lib}.c"
        rc, err = sh(["gcc", "-O2", "-o", str(exe), str(src), f"-l{lib}", "-lm"])
        if rc != 0:
            report["gcc_errors"][f"{lib}_{tag}"] = err[-2000:]
            return {"compiled": False}
        binp = WORK / f"{lib}_{tag}.bin"
        rc, err = sh([str(exe), "160", "120", "24", "28", str(offv), str(binp)],
                     timeout=600)
        if rc != 0:
            report["gcc_errors"][f"{lib}_{tag}_run"] = err[-2000:]
            return {"compiled": True, "ran": False, "stderr": err[-800:]}
        return {"compiled": True, "ran": True, "bytes": binp.stat().st_size}

    for lib, field_probe in (("x264", "x264_header_quant_offsets"),
                             ("x265", "x265_header_quantOffsets")):
        if report.get(field_probe, 0) > 0:
            base = sign_test(lib, 0.0, "base")
            entry = {"base": base}
            if base.get("ran"):
                for tag, offv in (("off_pos", 10.0), ("off_neg", -10.0)):
                    r = sign_test(lib, offv, tag)
                    entry[tag] = r
                    if r.get("ran"):
                        entry[f"ratio_{tag}"] = round(r["bytes"] / base["bytes"], 4)
                rp = entry.get("ratio_off_pos")
                rn = entry.get("ratio_off_neg")
                if rp is not None:
                    entry["works_and_sign_positive"] = bool(rp < 0.95 and (rn is None or rn > 1.05))
                    entry["verdict"] = ("C-API ROI path WORKS (pos saves, neg costs)"
                                        if entry["works_and_sign_positive"]
                                        else "offsets have NO effect in this build -> path DEAD")
            report["candidates"][lib] = entry
        else:
            report["candidates"][lib] = {"declared_in_header": False}


    # --- round 10: PATCHED library builds (owner-approved direction) ---
    # Root cause chain: stock builds gate the AQ/quant_offsets path; patch the
    # single gate line per library, rebuild, rerun the sign test.
    def patched_x265() -> dict:
        src = WORK / "x265src"
        if not src.exists():
            rc, err = sh(["git", "clone", "--depth", "1",
                          "https://github.com/videolan/x265.git", str(src)], timeout=900)
            if rc != 0:
                return {"error": err[-500:]}
        enc_cpp = src / "source" / "encoder" / "encoder.cpp"
        txt = enc_cpp.read_text(encoding="utf-8", errors="ignore")
        patched = txt.replace("m_param->rc.aqMode = X265_AQ_NONE;",
                              "m_param->rc.aqMode = X265_AQ_VARIANCE;")
        report["x265_patch_applied"] = txt != patched
        enc_cpp.write_text(patched, encoding="utf-8")
        b = src / "build"
        b.mkdir(exist_ok=True)
        rc, err = sh(["bash", "-c",
                      f"cd {b} && cmake ../source -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1; "
                      f"make -j4 > make.log 2>&1; echo rc=$?; find . -name 'libx265*' | head"], timeout=2400)
        report["x265_build_tail"] = err[-800:]
        libs = list(b.rglob("libx265.so"))
        if not libs:
            return {"error": "no libx265.so built", "log": err[-500:]}
        libdir = libs[0].parent
        exe = WORK / "hunt_x265p"
        rc, err = sh(["bash", "-c",
                      f"gcc -O2 -o {exe} {REPO}/kaggle/roi_hunt_x265.c "
                      f"-I{src}/source -L{libdir} -lx265 -lm"])
        if rc != 0:
            report["gcc_errors"]["x265p"] = err[-800:]
            return {"error": "compile failed", "log": err[-500:]}
        env = dict(os.environ, LD_LIBRARY_PATH=str(libdir))
        res = {}
        for tag, offv in (("base", 0.0), ("off", 10.0)):
            binp = WORK / f"x265p_{tag}.bin"
            p2 = subprocess.run([str(exe), "160", "120", "24", "28", str(offv), str(binp)],
                                capture_output=True, text=True, timeout=600, env=env)
            res[tag] = {"ran": p2.returncode == 0, "stderr": p2.stderr[-300:]}
            if p2.returncode == 0:
                res[tag]["bytes"] = binp.stat().st_size
        if res["base"].get("bytes") and res["off"].get("bytes"):
            res["ratio"] = round(res["off"]["bytes"] / res["base"]["bytes"], 4)
            res["works_and_sign_positive"] = bool(res["ratio"] < 0.95)
        return res

    try:
        report["candidates"]["x265_patched"] = patched_x265()
    except Exception as exc:  # noqa: BLE001
        report["candidates"]["x265_patched"] = {"error": str(exc)[:400]}

    try:
        report["candidates"]["x265_patched"] = report["candidates"].get("x265_patched", {})
    except Exception:
        pass

    try:
        report_write = json.dumps(report, indent=2)
        (OUT / "roi_hunt.json").write_text(report_write)
    except Exception:
        pass

    try:
        report["candidates"].pop("x265_patched_dup", None)
    except Exception:
        pass

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
