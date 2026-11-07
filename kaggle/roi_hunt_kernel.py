#!/usr/bin/env python
"""Kaggle kernel entry for the ROI hunt: run run_roi_hunt.sh, keep outputs lean."""
import json
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
t0 = time.time()
if not (WORK / "scope").exists():
    subprocess.run(["git", "clone", "https://github.com/wagur1/scope.git"],
                   cwd=WORK, capture_output=True, text=True)
proc = subprocess.run(["bash", "scope/kaggle/run_roi_hunt.sh"], cwd=WORK,
                      capture_output=True, text=True, timeout=5400)
(WORK / "roi_hunt_run.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
print(proc.stdout[-3000:])
print("rc", proc.returncode, "elapsed", round(time.time() - t0, 1))
