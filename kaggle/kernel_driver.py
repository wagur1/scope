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



def record_gpu(summary: dict) -> None:
    """Which GPU did the API actually assign? (metadata has no type field)"""
    try:
        import torch
        if torch.cuda.is_available():
            summary["gpu"] = [torch.cuda.get_device_name(i)
                              for i in range(torch.cuda.device_count())]
        else:
            summary["gpu"] = "none"
    except Exception as exc:
        summary["gpu"] = f"probe failed: {exc}"

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
    record_gpu(summary)
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

    # 3b. SCOPE integration (Bottleneck 2): clone upstream + fetch the
    # pretrained full-model checkpoint (Google Drive folder, per upstream
    # README). Graceful: A3 stays disabled if anything here fails.
    summary["scope"] = "not attempted"
    try:
        if not (WORK / "SCOPE").exists():
            run("clone_scope", ["git", "clone", "--depth", "1",
                                "https://github.com/miccunifi/SCOPE.git",
                                str(WORK / "SCOPE")], cwd=WORK, check=False, timeout=600)
        run("pip_gdown", ["python", "-m", "pip", "install", "-q", "gdown"], cwd=REPO,
            check=False, timeout=300)
        runs_dir = WORK / "SCOPE" / "runs"
        if not any(runs_dir.rglob("best_val.pt")) if runs_dir.exists() else True:
            run("scope_weights", ["python", "-m", "gdown", "--folder",
                                  "https://drive.google.com/drive/folders/1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES",
                                  "-O", str(runs_dir)], cwd=WORK, check=False, timeout=1500)
        ckpt = next(runs_dir.rglob("best_val.pt"), None) if runs_dir.exists() else None
        if ckpt:
            cfg["scope_repo"] = str(WORK / "SCOPE")
            cfg["scope_ckpt"] = str(ckpt)
            cfg["scope_family"] = "diff_jpeg"
            cfg["scope_severity"] = 0.5
            cfg["arms"]["A3_preedit_scope"] = True
            summary["scope"] = f"enabled A3, ckpt={ckpt.name}"
        else:
            summary["scope"] = "checkpoint not found — A3 disabled"
    except Exception as exc:  # noqa: BLE001 — smoke must survive SCOPE failures
        summary["scope"] = f"failed: {exc}"

    # 4. synthetic data + E1 grid. On toolchains without per-block QP map
    # support (stock ffmpeg x265 — probe above), the ROI arms are DISABLED in
    # the config: the loud-fail design stays intact for real runs, but the
    # smoke validates the rest of the pipeline end-to-end.
    run("gen_synth", ["python", "kaggle/gen_synthetic.py", str(SEED)], cwd=REPO)
    import yaml
    cfg_path = WORK / "e1_synth.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if not summary["x265_has_qp_delta_map_doc"]:
        cfg["arms"]["A1_roi"] = False
        cfg["arms"]["A4_preedit_fb_roi"] = False
        summary["roi_arms"] = "disabled (toolchain lacks qp-delta-map — expected on stock ffmpeg)"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run("run_e1", ["python", "scripts/run_e1.py", str(cfg_path)],
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

    # 6. lean output packaging: the kernel-output API truncates large file
    # trees (round 2 lost work_e1/results behind thousands of frames) — copy
    # ONLY the deliverables to out/ and remove every heavy directory.
    import shutil
    out = WORK / "out"
    out.mkdir(exist_ok=True)
    res = WORK / "work_e1" / "results"
    if res.exists():
        for f in res.glob("*.csv"):
            shutil.copy2(f, out / f.name)
        artifacts = {f.name: sum(1 for _ in f.open()) for f in res.glob("*.csv")}
    else:
        artifacts = {}
    summary["result_rows"] = artifacts
    for d in ("scope", "synth", "synth_gt", "synth_masks", "work_e1"):
        shutil.rmtree(WORK / d, ignore_errors=True)
    for f in WORK.glob("*.log"):  # stale root logs, if any
        f.unlink(missing_ok=True)
    summary["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
