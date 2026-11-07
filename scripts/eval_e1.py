#!/usr/bin/env python
"""E1 evaluation pass: fill the quality + ceiling columns the report needs.

Produces (resumable):
  results/quality.csv  seq, arm, qp, delta, quality   — mAP of each decoded arm
  results/ceiling.csv  seq, arm, ceiling              — mAP at negligible compression

Ceiling policy (MATH.md section 4, symmetric): A0 ceiling = mAP of the
ORIGINAL frames through the lossless path; content arms (A2/A3) = mAP of
their EDITED frames through the SAME lossless path. A1 has no own ceiling —
it inherits m*_anchor (non-modifying arm).

Usage:
  python scripts/eval_e1.py <config> [--weights yolov8n.pt]
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.codec import EncoderCfg, encode  # noqa: E402
from scope_vcm.detect import detect_frames, mAP  # noqa: E402

Q_FIELDS = ["seq", "arm", "qp", "delta", "quality"]
C_FIELDS = ["seq", "arm", "ceiling"]


def _rows(path: Path, fields: list[str]) -> tuple[list[dict], set]:
    rows, keys = [], set()
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
                keys.add((r["seq"], r["arm"]))
    return rows, keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--weights", required=True, help="frozen scorer checkpoint (same for all arms)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["workdir"]).resolve()
    gt_json = cfg["gt_json"]

    with (root / "results" / "e1.csv").open() as f:
        enc_rows = [r for r in csv.DictReader(f) if r["arm"] != ""]
    if not enc_rows:
        raise SystemExit("results/e1.csv empty — run run_e1.py first")

    q_path = root / "results" / "quality.csv"
    c_path = root / "results" / "ceiling.csv"
    q_rows, _ = _rows(q_path, Q_FIELDS)
    q_done = {(r["seq"], r["arm"], int(r["qp"]), int(r["delta"])) for r in q_rows}
    c_rows, c_done = _rows(c_path, C_FIELDS)
    fps = float(cfg.get("fps", 30.0))

    # frame count + image ids per sequence (GT ids assumed 0..T-1 unless mapped)
    seq_frames = {s["id"]: sorted(Path(s["frames"]).resolve().glob("*.png"))
                  for s in cfg["sequences"]}
    arms_cfg = cfg.get("arms", {})
    content_arms = [a for a in ("A2_preedit_fb", "A2c_preedit_blurall", "A3_preedit_scope")
                    if arms_cfg.get(a, a == "A2_preedit_fb")]

    # --- ceilings (lossless, symmetric path) --------------------------------
    for seq in cfg["sequences"]:
        sid = seq["id"]
        n = len(seq_frames[sid])
        if (sid, "A0_anchor") not in c_done:
            ll = encode(EncoderCfg(codec=cfg.get("encoder", "x265"), lossless=True,
                                   preset=cfg.get("preset", "medium")),
                        seq_frames[sid].parent, root / "enc" / "ceiling" / sid / "A0",
                        n_frames=n, fps=fps)
            ids = list(range(n))
            val = mAP(detect_frames(ll.decoded_dir, args.weights), gt_json, ids)
            c_rows.append({"seq": sid, "arm": "A0_anchor", "ceiling": round(val, 6)})
            c_done.add((sid, "A0_anchor"))
            logging.info("ceiling %s A0_anchor: %.4f", sid, val)
        for arm in content_arms:
            edited = root / "edited" / arm / sid
            if not edited.exists() or (sid, arm) in c_done:
                continue
            ll = encode(EncoderCfg(codec=cfg.get("encoder", "x265"), lossless=True,
                                   preset=cfg.get("preset", "medium")),
                        edited, root / "enc" / "ceiling" / sid / arm, n_frames=n, fps=fps)
            val = mAP(detect_frames(ll.decoded_dir, args.weights), gt_json, list(range(n)))
            c_rows.append({"seq": sid, "arm": arm, "ceiling": round(val, 6)})
            c_done.add((sid, arm))
            logging.info("ceiling %s %s: %.4f", sid, arm, val)

    # --- per-point quality ---------------------------------------------------
    for r in enc_rows:
        key = (r["seq"], r["arm"], int(r["qp"]), int(r["delta"]))
        if key in q_done:
            continue
        decoded = root / "enc" / r["arm"] / r["seq"] / f"qp{r['qp']}_d{r['delta']}" / "decoded"
        n = len(seq_frames[r["seq"]])
        val = mAP(detect_frames(decoded, args.weights), gt_json, list(range(n)))
        q_rows.append({"seq": r["seq"], "arm": r["arm"], "qp": r["qp"],
                       "delta": r["delta"], "quality": round(val, 6)})
        q_done.add(key)
        logging.info("quality %s %s qp%s d%s: %.4f",
                     r["seq"], r["arm"], r["qp"], r["delta"], val)

    with q_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=Q_FIELDS)
        w.writeheader()
        w.writerows(q_rows)
    with c_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=C_FIELDS)
        w.writeheader()
        w.writerows(c_rows)
    logging.info("eval pass done: %d quality rows, %d ceiling rows", len(q_rows), len(c_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
