#!/usr/bin/env python
"""E1 report: per-sequence BD, ceiling drops, gates G-eta / G-1, go/no-go verdict.

Inputs:
  results/e1.csv       seq, arm, qp, delta, rate_kbps          (encode pass)
  results/quality.csv  seq, arm, qp, delta, quality            (eval pass)
  results/ceiling.csv  seq, arm, ceiling                       (uncompressed accuracy)

Normalization policy (MATH.md section 4, pinned by the math-review pass):
  * content-modifying arms (A2/A3/A4) are normalized by their OWN ceiling m*_E
  * non-modifying arms (A0/A1) are normalized by m*_anchor — they share the
    anchor's content, so a second noisy ceiling measurement is pure variance
  * eta gate applies to the THESIS ARM ONLY (A2, or A3 when enabled) — A4 is
    diagnostic and never gates

ROI delta rule (pre-registered): the ROI baseline for sequence v is the
STRONGEST configured delta — g_roi_v = min over delta of BD_v(A0, A1(delta)).
Comparing against a weaker-than-best lossless baseline would inflate the gap
(math-review C1).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.bd import bd_rate, ceiling_drop  # noqa: E402
from scope_vcm.metrics import bootstrap_ci_median, thesis_test  # noqa: E402

CONTENT_ARMS = {"A2_preedit_fb", "A3_preedit_scope", "A4_preedit_fb_roi"}
THESIS_ARMS = ["A2_preedit_fb", "A3_preedit_scope"]  # first enabled one is the thesis arm


def load(workdir: Path):
    rates = pd.read_csv(workdir / "results" / "e1.csv")
    q = pd.read_csv(workdir / "results" / "quality.csv")
    cel = pd.read_csv(workdir / "results" / "ceiling.csv")
    merged = rates.merge(q, on=["seq", "arm", "qp", "delta"], how="inner")
    if len(merged) < len(rates):
        raise ValueError(
            f"{len(rates) - len(merged)} rate rows have no quality row — the eval "
            "pass is incomplete; refusing to report on silently-shrunk curves"
        )
    dup = cel.duplicated(subset=["seq", "arm"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (seq, arm) ceiling rows — clean ceiling.csv")
    return merged, cel


def normalized_points(s: pd.DataFrame, arm: str, ceilings: pd.DataFrame, sid: str):
    pts = s[s.arm == arm][["quality", "rate_kbps"]].values.astype(float)
    if len(pts) < 2:
        return None
    if (sid, arm) not in ceilings.index:
        raise ValueError(
            f"missing ceiling for ({sid}, {arm}) — add it to ceiling.csv; "
            "normalization is mandatory, never mix raw and normalized axes"
        )
    return pts / ceilings.loc[(sid, arm), "ceiling"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    args = ap.parse_args()
    workdir = Path(args.workdir)
    df, cel = load(workdir)
    df = df.copy()

    seqs = sorted(df.seq.unique())
    thesis_arm = next((a for a in THESIS_ARMS if a in set(df.arm)), None)
    if thesis_arm is None:
        raise SystemExit("no thesis arm (A2/A3) present in results — nothing to test")

    print(f"# E1 report — {len(seqs)} sequences, thesis arm: {thesis_arm}\n")

    # --- ceilings (gate G-eta, thesis arm only) ----------------------------
    anchor_ceil = {sid: float(cel.loc[(sid, "A0_anchor"), "ceiling"])
                   for sid in seqs if (sid, "A0_anchor") in cel.index}
    missing_anchor = [s for s in seqs if s not in anchor_ceil]
    if missing_anchor:
        raise ValueError(f"missing A0_anchor ceilings for {missing_anchor}")

    print("## Ceiling drops (eta), thesis arm only\n")
    print("| seq | arm | eta | flag(>0.01) |")
    print("|---|---|---|---|")
    etas = []
    for sid in seqs:
        if (sid, thesis_arm) not in cel.index:
            raise ValueError(f"missing ceiling for thesis arm ({sid}, {thesis_arm})")
        eta = ceiling_drop(float(cel.loc[(sid, thesis_arm), "ceiling"]), anchor_ceil[sid])
        etas.append(eta)
        print(f"| {sid} | {thesis_arm} | {eta:.4f} | {'FLAG' if eta > 0.01 else ''} |")

    # --- per-sequence BD (normalized quality) ------------------------------
    print("\n## BD-rate per sequence (ceiling-normalized quality axis)\n")
    print("| seq | pre-edit | ROI(best delta) | gap(pre - roi) |")
    print("|---|---|---|---|")
    gaps = []
    worst = (None, -np.inf)
    for sid in seqs:
        s = df[df.seq == sid].copy()
        # normalize per arm with the pinned policy (content arms: own ceiling;
        # A0/A1: anchor ceiling)
        for arm in s.arm.unique():
            ref = anchor_ceil[sid] if arm not in CONTENT_ARMS else \
                float(cel.loc[(sid, arm), "ceiling"])
            mask = s.arm == arm
            s.loc[mask, "quality"] = s.loc[mask, "quality"] / ref

        a = s[s.arm == "A0_anchor"][["quality", "rate_kbps"]].values.astype(float)
        if (sid, thesis_arm) not in cel.index:
            continue
        p_arm = s[s.arm == thesis_arm][["quality", "rate_kbps"]].values.astype(float)
        g_prop = bd_rate(a, p_arm) if len(p_arm) >= 2 else float("nan")

        # strongest configured lossless competitor (pre-registered rule)
        roi_bds = {}
        for d in sorted(s[s.arm == "A1_roi"].delta.unique()):
            p_roi = s[(s.arm == "A1_roi") & (s.delta == d)][["quality", "rate_kbps"]].values
            if len(p_roi) >= 2:
                roi_bds[int(d)] = bd_rate(a, p_roi)
        g_roi = min(roi_bds.values()) if roi_bds else float("nan")
        best_delta = min(roi_bds, key=roi_bds.get) if roi_bds else None

        gap = g_prop - g_roi
        gaps.append(gap)
        if np.isfinite(gap) and gap > worst[1]:
            worst = (sid, gap)
        print(f"| {sid} | {g_prop:.2f} | {g_roi:.2f} (d={best_delta}) | {gap:.2f} |")

    print(f"\nWorst-case gap: {worst[0]} ({worst[1]:.2f}%)  "
          f"(per-sequence reporting mandated by MATH.md section 5)")

    # --- gates --------------------------------------------------------------
    print("\n## Gates\n")
    tt = thesis_test(np.array(gaps))
    eta_ok = all(e <= 0.01 for e in etas)
    ci = bootstrap_ci_median(np.array(gaps)) if tt.get("n", 0) >= 8 else (float("nan"),) * 2
    print(f"- G-eta (thesis-arm ceiling drops <= 0.01): "
          f"{'PASS' if eta_ok else 'FAIL — F2 fires for the thesis arm'}")
    if "reason" in tt:
        print(f"- G-1: FAIL — {tt['reason']}")
    else:
        print(f"- G-1 (pre-edit beats strongest ROI delta-QP): "
              f"{'PASS' if tt['pass'] else 'FAIL — F1 fires'}")
        print(f"  median gap {tt['median_gap_pct']:.2f}%  "
              f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  "
              f"Wilcoxon p={tt['wilcoxon_p']:.4f}  sign-test p={tt['sign_test_p']:.4f}  "
              f"n={tt['n']}")
    go = tt.get("pass", False) and eta_ok
    print(f"\n**VERDICT: {'GO — premise alive, proceed to E2 additivity' if go else 'NO-GO — pivot to GAP-10 proxy-fidelity study; publish E1 as null + ROI baseline'}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
