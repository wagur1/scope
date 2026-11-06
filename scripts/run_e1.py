#!/usr/bin/env python
"""CLI: run the E1 encode/pre-edit grid. Evaluation (quality column) is a
separate pass so codec failures and detector failures are debuggable apart."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.experiment import run_e1  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="E1: pre-edit vs ROI delta-QP grid")
    ap.add_argument("config", help="YAML config, see configs/e1_decisive.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = run_e1(args.config)
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
