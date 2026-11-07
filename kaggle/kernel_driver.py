#!/usr/bin/env python
"""Kaggle kernel driver — E1 smoke run on a fresh environment.

Steps: clone repo -> pytest -> toolchain check -> synthetic E1 grid
(A0/A2/A2c/A4 real encodes; A1 attempted and its loud-fail RECORDED — on
stock ffmpeg it must fail, which settles the qp-delta-map question
empirically per toolchain) -> YOLO eval -> report -> summary.json.

Everything lands in /kaggle/working (kernel output). No secrets here.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "scope"
SEED = int(os.environ.get("SYNTH_SEED", "0"))
summary: dict = {"seed": SEED, "steps": {}}


def run(name: str, cmd: list[str], cwd: Path | None = None, check: bool = True,
        timeout: int | None = 3600) -> str:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    log = WORK / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    summary["steps"][name] = {
        "returncode": proc.returncode, "seconds": round(time.time() - t0, 1),
        "log": str(log.relative_to(WORK)),
    }
    print(f"[{name}] rc={proc.returncode} ({summary['steps'][name]['seconds']}s) -> {log}")
    if check and proc.returncode != 0:
        raise RuntimeError(f"step {name} failed, see {log}")
    return proc.stdout + proc.stderr


def main() -> None:
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)

    # 1. repo + deps
    if not REPO.exists():
        run("clone", ["git", "clone", "https://github.com/wagur1/scope.git"], cwd=WORK)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
        summary["git_sha"] = sha
    except Exception:
        summary["git_sha"] = "unknown"
    run("pip", ["python", "-m", "pip", "install", "-q", "-r",
                "requirements.txt", "ultralytics"], cwd=REPO)

    # 2. sanity: BD tests on a fresh env
    out = run("pytest", ["python", "-m", "pytest", "tests/", "-q"], cwd=REPO)
    summary["pytest_tail"] = out.strip().splitlines()[-1] if out.strip() else "no output"

    # 3. toolchain: ffmpeg version + encoder probe + ROI map self-test
    ver = run("ffmpeg_version", ["ffmpeg", "-version"])
    summary["ffmpeg_version"] = ver.splitlines()[0]
    probe = run("x265_probe", ["ffmpeg", "-h", "encoder=libx265"], check=False)
    summary["x265_has_qp_delta_map_doc"] = "qp-delta-map" in probe.lower()
    run("check_encoders", ["python", "scripts/check_encoders.py"], cwd=REPO, check=False)

    # 4. synthetic data + E1 grid (A1 attempted; loud-fail is the expected,
    #    recorded outcome on stock toolchains)
    run("gen_synth", ["python", "kaggle/gen_synthetic.py", str(SEED)], cwd=REPO)
    run("run_e1", ["python", "scripts/run_e1.py", str(WORK / "e1_synth.yaml")],
        cwd=REPO, check=False)

    # 5. eval + report (detector downloads yolov8n.pt — internet required)
    run("eval_e1", ["python", "scripts/eval_e1.py", str(WORK / "e1_synth.yaml"),
                    "--weights", "yolov8n.pt"], cwd=REPO, check=False)
    rep = run("report_e1", ["python", "scripts/report_e1.py", str(WORK / "work_e1"),
                            "--config", str(WORK / "e1_synth.yaml")],
              cwd=REPO, check=False)
    (WORK / "report.md").write_text(rep)
    m = re.search(r"\*\*VERDICT: (.+?)\*\*", rep)
    summary["verdict"] = m.group(1) if m else "verdict line not found"

    # 6. collect artifacts
    res = WORK / "work_e1" / "results"
    artifacts = {}
    if res.exists():
        for f in res.glob("*.csv"):
            artifacts[f.name] = sum(1 for _ in f.open())
    summary["result_rows"] = artifacts
    summary["elapsed_s"] = round(time.time() - t_start, 1)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
