#!/usr/bin/env python
"""ROI hunt round 15: R15a (kvazaar source build + --roi sign test) and
R15b (x265 master grep for per-CU QP map support). Sequential-route step 1.

Output: out/roi_hunt15.json
"""
import json
import os
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"
report: dict = {"r15a_kvazaar": {}, "r15b_x265_master": {}}


def sh(cmd: list, timeout: int = 1200) -> tuple:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)

    if not (WORK / "scope").exists():
        sh(["git", "clone", "https://github.com/wagur1/scope.git", str(WORK / "scope")])
    REPO = WORK / "scope"

    # ---------------- R15a: kvazaar from source ----------------
    def kvazaar() -> dict:
        src = WORK / "kvazaarsrc"
        if not src.exists():
            rc, err = sh(["git", "clone", "--depth", "1",
                          "https://github.com/ultravideo/kvazaar.git", str(src)], timeout=900)
            if rc != 0:
                return {"error": err[-400:]}
        rc, err = sh(["bash", "-c",
                      f"cd {src} && cmake -B build -DCMAKE_BUILD_TYPE=Release "
                      f"> cmake.log 2>&1 && cmake --build build -j4 > make.log 2>&1; "
                      f"find build -name 'kvazaar*' -type f | head"], timeout=2400)
        res = {}
        # round 15d fix: probe every executable for the real CLI (the tests
        # binary also matches *kvazaar* but rejects --input); pick the one
        # whose --help mentions --input
        exe = None
        if (src / "build").exists():
            cands = [c for c in (src / "build").rglob("*kvazaar*")
                     if c.is_file() and c.suffix == "" and os.access(c, os.X_OK)]
            for c in cands:
                r3 = subprocess.run([str(c), "--help"], capture_output=True,
                                    text=True, timeout=60)
                if "--input" in (r3.stdout + r3.stderr):
                    exe = c
                    res["cli_probe"] = f"{c.name}: CLI identified"
                    break
            else:
                res["cli_probe"] = f"no CLI among {[c.name for c in cands]}"
        if exe is None:
            return res
        if exe is None:
            return {"error": "kvazaar binary not built", "log": err[-500:]}
        ver = sh([str(exe), "--version"])[1][:60]

        w, h, n = 160, 120, 24
        yuvp = WORK / "kvz.yuv"
        # deterministic clip, same generator as prior rounds
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
        yuvp.write_bytes(buf)

        res = {"version": ver}
        # ROI map format (upstream README): first line "blocks_w blocks_h",
        # then blocks_w*blocks_h QP values per frame, 8x8 granularity
        mb = 8
        mbw, mbh = -(-w // mb), -(-h // mb)
        for tag, offv in (("base", None), ("off", 10)):
            outp = WORK / f"kvz15_{tag}.hevc"
            cmd = [str(exe), "--input", str(yuvp), "--input-res", f"{w}x{h}",
                   "--input-fps", "30", "--output", str(outp), "--qp", "28",
                   "--threads", "1", "--no-aq"]
            if offv is not None:
                roip = WORK / f"kvz15_roi_{tag}.txt"
                lines = [f"{mbw} {mbh}"]
                for _t in range(n):
                    lines.append(" ".join([str(28 + int(offv))] * (mbw * mbh)))
                roip.write_text(chr(10).join(lines) + chr(10))
                cmd += ["--roi", str(roip)]
            r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            res[tag] = {"ran": r2.returncode == 0 and outp.exists(),
                        "rc": r2.returncode,
                        "stderr": r2.stderr[-300:], "stdout": r2.stdout[-300:]}
            if res[tag]["ran"]:
                res[tag]["bytes"] = outp.stat().st_size
        if res.get("base", {}).get("bytes") and res.get("off", {}).get("bytes"):
            res["ratio"] = round(res["off"]["bytes"] / res["base"]["bytes"], 4)
            res["works_and_sign_positive"] = bool(res["ratio"] < 0.95)
        return res

    try:
        report["r15a_kvazaar"] = kvazaar()
    except Exception as exc:  # noqa: BLE001
        report["r15a_kvazaar"] = {"error": str(exc)[:400]}

    # ---------------- R15b: x265 master grep ----------------
    def x265_master() -> dict:
        src = WORK / "x265master"
        if not src.exists():
            sh(["git", "clone", "--depth", "1",
                "https://github.com/videolan/x265.git", str(src)], timeout=900)
        hits = {}
        for pat in ("qp-delta-map", "qpDeltaMap", "quantOffsets", "RegionOfInterest",
                    "roiMap", "per-cu qp"):
            cnt = 0
            files = []
            for f in (src / "source").rglob("*"):
                if f.suffix in (".h", ".cpp", ".c") and f.is_file():
                    try:
                        t = f.read_text(encoding="utf-8", errors="ignore")
                        c = t.lower().count(pat.lower())
                        if c:
                            cnt += c
                            files.append(f.name)
                    except Exception:
                        pass
            hits[pat] = {"count": cnt, "files": sorted(set(files))[:6]}
        # x265 CLI options mentioning roi or delta
        rc, out = sh(["bash", "-c",
                      f"grep -rn 'roi\\|delta' {src}/source/x265.h | head -12"], timeout=120)
        hits["x265h_lines"] = out.strip().splitlines()[:12]
        return hits

    try:
        report["r15b_x265_master"] = x265_master()
    except Exception as exc:  # noqa: BLE001
        report["r15b_x265_master"] = {"error": str(exc)[:400]}

    report["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "roi_hunt15.json").write_text(json.dumps(report, indent=2))
    # strip heavy trees (output API truncates large file listings — round 15
    # first attempt lost out/ behind kvazaarsrc/.git)
    import shutil
    for d in ("kvazaarsrc", "x265master", "scope", "kvz.yuv"):
        shutil.rmtree(WORK / d, ignore_errors=True)
    for f in WORK.glob("kvz15_*"):
        f.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
