#!/usr/bin/env python
"""ROI hunt — Bottleneck 1 (A1 unblock). Round 12 final layout.

Tracks:
  1. stock x264 C-API sign test          (rounds 5-9: confirmed DEAD on Ubuntu build)
  2. stock x265 C-API sign test          (crashes in container harness)
  3. patched x265 (AQ_NONE gate -> VARIANCE, cmake shared build)
  4. patched x264 (CQP->AQ gate patch, static build)
  5. kvazaar stock CLI --roi             (HEVC, zero patching — Argon rank 3)
Output: out/roi_hunt.json
"""
import json
import os
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "scope"
OUT = WORK / "out"
report: dict = {"candidates": {}, "gcc_errors": {}}


def sh(cmd: list, timeout: int = 900) -> tuple:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def subprocess_run(cmd: list, env: dict, timeout: int) -> tuple:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout + p.stderr


def make_yuv(path: Path, w: int, h: int, n: int) -> None:
    import random as _r
    _r.seed(7)
    buf = bytearray()
    for t in range(n):
        y = bytes([60 + _r.randrange(140) for _ in range(w * h)])
        u = bytes([128]) * (w // 2 * h // 2)
        v = bytes([128]) * (w // 2 * h // 2)
        frame = bytearray(y + u + v)
        bx = (t * 2) % (w - 64)
        by = (t * 3) % (h - 64)
        for yy in range(by, by + 48):
            for xx in range(bx, bx + 48):
                frame[yy * w + xx] = 235
        buf += frame
    path.write_bytes(buf)


def sign_test_c(lib: str, offv: float, tag: str) -> dict:
    """Compile roi_hunt_<lib>.c (system headers/libs) and encode with offset."""
    exe = WORK / f"hunt_{lib}_{tag}"
    src = REPO / "kaggle" / f"roi_hunt_{lib}.c"
    rc, err = sh(["gcc", "-O2", "-o", str(exe), str(src), f"-l{lib}", "-lm"])
    if rc != 0:
        report["gcc_errors"][f"{lib}_{tag}"] = err[-1500:]
        return {"compiled": False, "stderr": err[-400:]}
    binp = WORK / f"{lib}_{tag}.bin"
    rc, out = subprocess_run([str(exe), "160", "120", "24", "28", str(offv), str(binp)],
                             dict(os.environ), 600)
    if rc != 0:
        return {"compiled": True, "ran": False, "stderr": out[-400:]}
    return {"compiled": True, "ran": True, "bytes": binp.stat().st_size}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)

    if not REPO.exists():
        sh(["git", "clone", "https://github.com/wagur1/scope.git", str(REPO)])

    # round 13: SEPARATE transactions, verify between steps (round 12: one
    # transaction adding kvazaar silently removed libx264-dev)
    sh(["bash", "-c", "apt-get update -qq && apt-get install -y -qq "
                      "build-essential cmake git >/dev/null 2>&1"], timeout=1200)
    sh(["bash", "-c", "apt-get install -y -qq libx264-dev >/dev/null 2>&1"], timeout=600)
    report["env_x264_h"] = Path("/usr/include/x264.h").exists()
    sh(["bash", "-c", "apt-get install -y -qq libx265-dev >/dev/null 2>&1"], timeout=600)
    report["env_x265_h"] = Path("/usr/include/x265.h").exists()
    kv = sh(["bash", "-c", "apt-get install -y -qq kvazaar >/dev/null 2>&1; which kvazaar"],
            timeout=600)
    report["env_kvazaar"] = kv[1].strip()[-40:]

    # ---- 1+2. stock C-API sign tests ----
    for lib, probe in (("x264", "x264_header_quant_offsets"),
                       ("x265", "x265_header_quantOffsets")):
        h = next(Path("/usr/include").rglob(f"{lib}.h"), None)
        report[f"{lib}_header_probe"] = h.read_text().count(
            "quant_offsets" if lib == "x264" else "quantOffsets") if h else 0
        if report[f"{lib}_header_probe"] > 0:
            base = sign_test_c(lib, 0.0, "base")
            entry = {"base": base}
            if base.get("ran"):
                pos = sign_test_c(lib, 10.0, "off_pos")
                entry["off_pos"] = pos
                if pos.get("ran"):
                    entry["ratio_off_pos"] = round(pos["bytes"] / base["bytes"], 4)
            report["candidates"][lib] = entry
        else:
            report["candidates"][lib] = {"declared_in_header": False}

    # ---- 3. patched x265 (AQ gate -> VARIANCE, shared build) ----
    def patched_x265() -> dict:
        src = WORK / "x265src"
        if not src.exists():
            rc, err = sh(["git", "clone", "--depth", "1",
                          "https://github.com/videolan/x265.git", str(src)], timeout=900)
            if rc != 0:
                return {"error": err[-400:]}
        import re as _re
        gate_hits = []
        for cpp in (src / "source").rglob("*.cpp"):
            t = cpp.read_text(encoding="utf-8", errors="ignore")
            for m in _re.finditer(r".*rc\.aqMode = X265_AQ_NONE.*", t):
                gate_hits.append((cpp, m.group(0).strip()))
        report["x265_gate_hits"] = [f"{c.name}: {l[:80]}" for c, l in gate_hits][:6]
        if not gate_hits:
            return {"error": "no rc.aqMode = X265_AQ_NONE gate found"}
        gate_cpp, gate_line = gate_hits[0]
        txt = gate_cpp.read_text(encoding="utf-8", errors="ignore")
        gate_cpp.write_text(txt.replace(gate_line.strip(),
                             gate_line.strip().replace("X265_AQ_NONE", "X265_AQ_VARIANCE")),
                            encoding="utf-8")
        report["x265_patch_applied"] = True
        b = src / "build"
        b.mkdir(exist_ok=True)
        rc, err = sh(["bash", "-c",
                      f"cd {b} && cmake ../source -DCMAKE_BUILD_TYPE=Release "
                      f"-DBUILD_SHARED_LIBS=ON -DENABLE_CLI=OFF > cmake.log 2>&1 && "
                      f"make -j4 > make.log 2>&1; echo done"], timeout=2400)
        libs = list(b.rglob("libx265.so"))
        if not libs:
            report["x265_build_tail"] = err[-700:]
            return {"error": "no libx265.so built"}
        libdir = libs[0].parent
        exe = WORK / "hunt_x265p"
        rc, err = sh(["gcc", "-O2", "-o", str(exe), str(REPO / "kaggle" / "roi_hunt_x265.c"),
                      f"-I{b}", f"-I{src}/source", f"-L{libdir}", "-lx265", "-lm"])
        if rc != 0:
            report["gcc_errors"]["x265p"] = err[-1200:]
            return {"error": "compile failed", "stderr": err[-400:]}
        env = dict(os.environ, LD_LIBRARY_PATH=str(libdir))
        res = {}
        for tag, offv in (("base", 0.0), ("off", 10.0)):
            binp = WORK / f"x265p_{tag}.bin"
            rc, out = subprocess_run([str(exe), "160", "120", "24", "28", str(offv), str(binp)],
                                     env, 600)
            res[tag] = {"ran": rc == 0, "stderr": out[-300:]}
            if rc == 0:
                res[tag]["bytes"] = binp.stat().st_size
        if res["base"].get("bytes") and res["off"].get("bytes"):
            res["ratio"] = round(res["off"]["bytes"] / res["base"]["bytes"], 4)
            res["works_and_sign_positive"] = bool(res["ratio"] < 0.95)
        return res

    # ---- 4. patched x264 (CQP->AQ gate patch, static build) ----
    def patched_x264() -> dict:
        src = WORK / "x264src"
        if not src.exists():
            rc, err = sh(["git", "clone", "--depth", "1",
                          "https://code.videolan.org/videolan/x264.git", str(src)], timeout=900)
            if rc != 0:
                return {"error": err[-400:]}
        import re as _re
        gate_hits = []
        for c_file in (src / "encoder").rglob("*.c"):
            t = c_file.read_text(encoding="utf-8", errors="ignore")
            for m in _re.finditer(r".*i_aq_mode = X264_AQ_NONE.*", t):
                gate_hits.append((c_file, m.group(0).strip()))
        report["x264_gate_hits"] = [f"{c.name}: {l[:80]}" for c, l in gate_hits][:6]
        if not gate_hits:
            return {"error": "no i_aq_mode = X264_AQ_NONE gate found"}
        gate_c, gate_line = gate_hits[0]
        txt = gate_c.read_text(encoding="utf-8", errors="ignore")
        gate_c.write_text(txt.replace(gate_line.strip(),
                           gate_line.strip().replace("X264_AQ_NONE", "X264_AQ_VARIANCE")),
                          encoding="utf-8")
        report["x264_patch_applied"] = True
        rc, err = sh(["bash", "-c",
                      f"cd {src} && ./configure --enable-static --enable-pic "
                      f"--disable-cli > configure.log 2>&1 && make -j4 > make.log 2>&1; "
                      f"echo done"], timeout=1800)
        liba = src / "libx264.a"
        if not liba.exists():
            report["x264_build_tail"] = err[-700:]
            return {"error": "build failed"}
        exe = WORK / "hunt_x264p"
        rc, err = sh(["gcc", "-O2", "-o", str(exe), str(REPO / "kaggle" / "roi_hunt_x264.c"),
                      str(liba), "-lm", "-lpthread"])
        if rc != 0:
            report["gcc_errors"]["x264p"] = err[-1200:]
            return {"error": "compile failed"}
        res = {}
        for tag, offv in (("base", 0.0), ("off", 10.0)):
            binp = WORK / f"x264p_{tag}.bin"
            rc, out = subprocess_run([str(exe), "160", "120", "24", "28", str(offv), str(binp)],
                                     dict(os.environ), 600)
            res[tag] = {"ran": rc == 0, "stderr": out[-300:]}
            if rc == 0:
                res[tag]["bytes"] = binp.stat().st_size
        if res["base"].get("bytes") and res["off"].get("bytes"):
            res["ratio"] = round(res["off"]["bytes"] / res["base"]["bytes"], 4)
            res["works_and_sign_positive"] = bool(res["ratio"] < 0.95)
        return res

    # ---- 5. kvazaar stock CLI --roi ----
    def kvazaar_roi() -> dict:
        ver = sh(["kvazaar", "--version"])[1][:60]
        w, h, n = 160, 120, 24
        yuvp = WORK / "kvz.yuv"
        make_yuv(yuvp, w, h, n)
        res = {"version": ver}
        for tag, offv in (("base", None), ("off", 10)):
            outp = WORK / f"kvz_{tag}.hevc"
            cmd = ["kvazaar", "-i", str(yuvp), "--input-res", f"{w}x{h}",
                   "--input-fps", "30", "-o", str(outp), "--qp", "28",
                   "--threads", "1", "--no-aq"]
            if offv is not None:
                mb = 8
                mbw, mbh = -(-w // mb), -(-h // mb)
                lines = [" ".join([str(int(offv))] * (mbw * mbh)) for _ in range(n)]
                (WORK / f"kvz_roi_{tag}.txt").write_text(chr(10).join(lines) + chr(10))
                cmd += ["--roi", str(WORK / f"kvz_roi_{tag}.txt")]
            r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            res[tag] = {"ran": r2.returncode == 0 and outp.exists(),
                        "stderr": r2.stderr[-300:]}
            if res[tag]["ran"]:
                res[tag]["bytes"] = outp.stat().st_size
        if res.get("base", {}).get("bytes") and res.get("off", {}).get("bytes"):
            res["ratio"] = round(res["off"]["bytes"] / res["base"]["bytes"], 4)
            res["works_and_sign_positive"] = bool(res["ratio"] < 0.95)
        return res

    def x264_master() -> dict:
        """x264 MASTER build: the CQP->AQ_NONE gate was removed upstream, so
        quant_offsets may now work where the old apt build ignores them."""
        src = WORK / "x264master"
        if not src.exists():
            rc, err = sh(["git", "clone", "--depth", "1",
                          "https://code.videolan.org/videolan/x264.git", str(src)], timeout=900)
            if rc != 0:
                return {"error": err[-400:]}
        report["x264_master_gate_present"] = "X264_AQ_NONE" in (
            (src / "encoder" / "encoder.c").read_text(encoding="utf-8", errors="ignore"))
        rc, err = sh(["bash", "-c",
                      f"cd {src} && ./configure --enable-static --enable-pic "
                      f"--disable-cli > configure.log 2>&1 && make -j4 > make.log 2>&1; "
                      f"echo done"], timeout=1800)
        liba = src / "libx264.a"
        if not liba.exists():
            report["x264_master_build_tail"] = err[-500:]
            return {"error": "build failed"}
        exe = WORK / "hunt_x264m"
        rc, err = sh(["gcc", "-O2", "-o", str(exe), str(REPO / "kaggle" / "roi_hunt_x264.c"),
                      str(liba), "-lm", "-lpthread"])
        if rc != 0:
            report["gcc_errors"]["x264m"] = err[-1000:]
            return {"error": "compile failed"}
        res = {}
        for tag, offv in (("base", 0.0), ("off", 10.0)):
            binp = WORK / f"x264m_{tag}.bin"
            rc, out = subprocess_run([str(exe), "160", "120", "24", "28", str(offv), str(binp)],
                                     dict(os.environ), 600)
            res[tag] = {"ran": rc == 0, "stderr": out[-300:]}
            if rc == 0:
                res[tag]["bytes"] = binp.stat().st_size
        if res["base"].get("bytes") and res["off"].get("bytes"):
            res["ratio"] = round(res["off"]["bytes"] / res["base"]["bytes"], 4)
            res["works_and_sign_positive"] = bool(res["ratio"] < 0.95)
        return res

    for tag, fn in (("x265_patched", patched_x265), ("x264_patched", patched_x264),
                    ("x264_master", x264_master),
                    ("kvazaar", kvazaar_roi)):
        try:
            report["candidates"][tag] = fn()
        except Exception as exc:  # noqa: BLE001
            report["candidates"][tag] = {"error": str(exc)[:400]}

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
