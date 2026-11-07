#!/usr/bin/env python
"""E1 report: per-sequence BD, ceiling drops, gates G-eta / G-1, verdict.

Verdict taxonomy (external review P1-03 — a failed demonstration is NOT a
falsified premise):
  INCOMPLETE    — a required input is missing (no A1, n < minimum, NaN BD,
                  missing ceilings/curves). No scientific conclusion permitted.
  INCONCLUSIVE  — valid comparison ran but the statistic did not clear the
                  gate (power/precision question, not a falsification).
  NO-SUPPORT    — valid comparison ran and failed: the tested arm does not
                  beat the strongest lossless ROI baseline.
  GO            — at least one co-primary arm passes G-eta AND G-1.

Co-primary policy (MATH.md section 6, external review P1-02): EVERY present
thesis arm (A2, A3) is evaluated independently; GO requires any arm to pass
both gates; F1 for a specific arm is scoped to that arm's method.

QP policy (owner decision, external review P1-06): headline analysis uses
primary_qps from the config (default the 5 VCM CTC levels); other collected
QPs are reported as a separate sensitivity table, never in the headline.

Ceiling inheritance: A4 encodes the SAME edited frames as A2 — if eval did
not emit an A4 ceiling, the report inherits A2's, with a printed note
(external review P1-05).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.bd import bd_rate, bd_rate_truncated, ceiling_drop  # noqa: E402
from scope_vcm.metrics import bootstrap_ci_median, thesis_test  # noqa: E402

CONTENT_ARMS = {"A2_preedit_fb", "A2c_preedit_blurall", "A3_preedit_scope", "A4_preedit_fb_roi"}
THESIS_ARMS = ["A2_preedit_fb", "A3_preedit_scope"]
KEY = ["seq", "arm", "qp", "delta"]
DEFAULT_PRIMARY_QPS = [22, 27, 32, 37, 42]


def load(workdir: Path, primary_qps: list[int] | None):
    """Load and join the three CSVs with strict schema checks (review P1-01, P2-01).

    e1.csv carries a placeholder quality column from the encode pass — it is
    dropped BEFORE the merge so the eval-pass quality is the single source.
    The merge is validated one-to-one on the full key, and the key sets are
    compared exactly (row counts alone let duplicates mask missing keys).
    """
    rates = pd.read_csv(workdir / "results" / "e1.csv").drop(
        columns=["quality", "is_ceiling", "notes"], errors="ignore")
    q = pd.read_csv(workdir / "results" / "quality.csv")
    cel = pd.read_csv(workdir / "results" / "ceiling.csv")

    for name, t in (("e1.csv", rates), ("quality.csv", q)):
        dup = t.duplicated(subset=KEY).sum()
        if dup:
            raise ValueError(f"{name}: {dup} duplicate {KEY} keys — clean the CSV")
    missing_keys = set(map(tuple, rates[KEY].values)) - set(map(tuple, q[KEY].values))
    if missing_keys:
        raise ValueError(
            f"{len(missing_keys)} rate rows have no quality row (e.g. "
            f"{sorted(missing_keys)[:3]}) — the eval pass is incomplete; refusing "
            "to report on a data-dependent subset")
    merged = rates.merge(q, on=KEY, how="inner", validate="one_to_one")

    dup = cel.duplicated(subset=["seq", "arm"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (seq, arm) ceiling rows — clean ceiling.csv")

    if primary_qps:
        before = len(merged)
        merged = merged[merged.qp.isin(primary_qps)]
        print(f"[qp-policy] headline analysis restricted to primary QPs "
              f"{primary_qps}: {before} -> {len(merged)} rows "
              f"(excluded rows are sensitivity-only, MATH.md section 6)")
        if merged.empty:
            raise ValueError("no rows left after primary-QP filter — check the config")
    return merged, cel.set_index(["seq", "arm"])


def ceiling_for(cel: pd.DataFrame, sid: str, arm: str):
    """Ceiling lookup with the A4->A2 inheritance rule (review P1-05)."""
    if (sid, arm) in cel.index:
        return float(cel.loc[(sid, arm), "ceiling"]), ""
    if arm == "A4_preedit_fb_roi" and (sid, "A2_preedit_fb") in cel.index:
        v = float(cel.loc[(sid, "A2_preedit_fb"), "ceiling"])
        return v, " (inherited from A2: identical edited source)"
    return None, " MISSING"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--config", help="config YAML: primary_qps + per-domain breakdown")
    args = ap.parse_args()
    workdir = Path(args.workdir)
    primary_qps = None
    domain_map: dict = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        primary_qps = cfg.get("primary_qps")
        for s in cfg.get("sequences", []):
            domain_map[s["id"]] = s.get("domain", "unspecified")

    df, cel = load(workdir, primary_qps)
    df = df.copy()
    seqs = sorted(df.seq.unique())
    present_thesis_arms = [a for a in THESIS_ARMS if a in set(df.arm)]
    if not present_thesis_arms:
        raise SystemExit("no thesis arm (A2/A3) present in results — nothing to test")

    anchor_ceil = {}
    for sid in seqs:
        v, note = ceiling_for(cel, sid, "A0_anchor")
        if v is None:
            raise ValueError(f"missing A0_anchor ceiling for {sid}")
        anchor_ceil[sid] = v

    print(f"# E1 report — {len(seqs)} sequences, co-primary arms: "
          f"{', '.join(present_thesis_arms)}\n")

    # --- eta per thesis arm -------------------------------------------------
    print("## Ceiling drops (eta), per thesis arm\n")
    print("| seq | arm | eta | flag(>0.01) |")
    print("|---|---|---|---|")
    etas_by_arm: dict = {a: [] for a in present_thesis_arms}
    for t_arm in present_thesis_arms:
        for sid in seqs:
            v, note = ceiling_for(cel, sid, t_arm)
            if v is None:
                raise ValueError(f"missing ceiling for thesis arm ({sid}, {t_arm})")
            eta = ceiling_drop(v, anchor_ceil[sid])
            etas_by_arm[t_arm].append(eta)
            print(f"| {sid} | {t_arm} | {eta:.4f}{note} | {'FLAG' if eta > 0.01 else ''} |")

    # --- per-arm per-sequence BD --------------------------------------------
    arm_gaps: dict = {a: {} for a in present_thesis_arms}   # {sid: gap} — keyed
    arm_incomplete: dict = {a: [] for a in present_thesis_arms}
    for t_arm in present_thesis_arms:
        print(f"\n## BD-rate per sequence — {t_arm} (ceiling-normalized + "
              "secondary absolute-mAP-capped)\n")
        print("| seq | domain | pre-edit (norm) | ROI best (norm) | gap (norm) | "
              "pre-edit (abs-cap) | gap (abs-cap) |")
        print("|---|---|---|---|---|---|---|")
        worst = (None, -np.inf)
        for sid in seqs:
            s_raw = df[df.seq == sid].copy()
            s = s_raw.copy()
            for arm in s.arm.unique():
                v, _ = ceiling_for(cel, sid, arm)
                if v is None:
                    arm_incomplete[t_arm].append(f"{sid}: missing ceiling {arm}")
                    continue
                ref = anchor_ceil[sid] if arm not in CONTENT_ARMS else v
                mask = s.arm == arm
                s.loc[mask, "quality"] = s.loc[mask, "quality"] / ref

            a = s[s.arm == "A0_anchor"][["quality", "rate_kbps"]].values.astype(float)
            p_arm = s[s.arm == t_arm][["quality", "rate_kbps"]].values.astype(float)
            if len(a) < 2 or len(p_arm) < 2:
                arm_incomplete[t_arm].append(f"{sid}: <2 RD points")
                continue
            g_prop = bd_rate(a, p_arm)

            q_cap = min(float(cel.loc[(sid, t_arm), "ceiling"]) if (sid, t_arm) in cel.index
                        else anchor_ceil[sid], anchor_ceil[sid])
            a_raw = s_raw[s_raw.arm == "A0_anchor"][["quality", "rate_kbps"]].values.astype(float)
            p_raw = s_raw[s_raw.arm == t_arm][["quality", "rate_kbps"]].values.astype(float)
            g_prop_abs = bd_rate_truncated(a_raw, p_raw, q_cap)

            roi_bds, roi_bds_abs = {}, {}
            for d in sorted(s[s.arm == "A1_roi"].delta.unique()):
                p_roi = s[(s.arm == "A1_roi") & (s.delta == d)][
                    ["quality", "rate_kbps"]].values
                p_roi_raw = s_raw[(s_raw.arm == "A1_roi") & (s_raw.delta == d)][
                    ["quality", "rate_kbps"]].values
                if len(p_roi) >= 2:
                    v = bd_rate(a, p_roi)
                    if np.isfinite(v):
                        roi_bds[int(d)] = v
                        roi_bds_abs[int(d)] = bd_rate_truncated(a_raw, p_roi_raw, q_cap)
            if not roi_bds:
                arm_incomplete[t_arm].append(f"{sid}: no finite ROI competitor BD")
                continue
            g_roi = min(roi_bds.values())
            best_delta = min(roi_bds, key=roi_bds.get)
            g_roi_abs = roi_bds_abs[best_delta]

            gap = g_prop - g_roi
            abs_gap = g_prop_abs - g_roi_abs
            arm_gaps[t_arm][sid] = gap
            if np.isfinite(gap) and gap > worst[1]:
                worst = (sid, gap)
            dom = domain_map.get(sid, "-")
            print(f"| {sid} | {dom} | {g_prop:.2f} | {g_roi:.2f} (d={best_delta}) | "
                  f"{gap:.2f} | {g_prop_abs:.2f} | {abs_gap:.2f} |")
        print(f"\nWorst-case normalized gap ({t_arm}): {worst[0]} ({worst[1]:.2f}%)")

    if domain_map:
        print("\n## Per-domain medians (MATH.md section 6)\n")
        for t_arm in present_thesis_arms:
            for dom in sorted(set(domain_map.values())):
                d_gaps = [g for s_id, g in arm_gaps[t_arm].items()
                          if domain_map.get(s_id) == dom and np.isfinite(g)]
                if d_gaps:
                    print(f"- {t_arm} / {dom} (n={len(d_gaps)}): "
                          f"median normalized gap = {float(np.median(d_gaps)):.2f}%")

    # --- gates, per arm, then the combined verdict ---------------------------
    # Taxonomy (external review P1-03 + Argon verify round):
    #   PASS / F2-FAIL / NO-SUPPORT / INCONCLUSIVE / INCOMPLETE — see MATH.md;
    # missing A1 or n<8 is INCOMPLETE and must never print NO-GO/publish-F1.
    print("\n## Gates (per co-primary arm)\n")
    arm_results: dict = {}
    for t_arm in present_thesis_arms:
        eta_ok = all(e <= 0.01 + 1e-12 for e in etas_by_arm[t_arm]) \
            if etas_by_arm[t_arm] else False
        gaps = arm_gaps[t_arm]
        if arm_incomplete[t_arm]:
            arm_results[t_arm] = ("INCOMPLETE", "; ".join(arm_incomplete[t_arm][:3]))
            print(f"- {t_arm}: INCOMPLETE — {arm_incomplete[t_arm][0]}")
            continue
        tt = thesis_test(np.array(list(gaps.values())))
        if "reason" in tt:
            arm_results[t_arm] = ("INCOMPLETE", tt["reason"])
            print(f"- {t_arm}: INCOMPLETE — {tt['reason']}")
            continue
        ci = bootstrap_ci_median(np.array(list(gaps.values())))
        med = tt["median_gap_pct"]
        g1_pass = tt["pass"]
        if g1_pass and eta_ok:
            arm_results[t_arm] = ("PASS", f"median {med:.2f}%")
            print(f"- {t_arm}: PASS — median gap {med:.2f}%  "
                  f"CI [{ci[0]:.2f}, {ci[1]:.2f}]  Wilcoxon p={tt['wilcoxon_p']:.4f}  "
                  f"sign p={tt['sign_test_p']:.4f}  n={tt['n']}")
        elif g1_pass and not eta_ok:
            arm_results[t_arm] = ("F2-FAIL", f"median {med:.2f}%, eta gate failed")
            print(f"- {t_arm}: F2-FAIL — G-1 passed but G-eta failed "
                  "(information-loss gate bounds the honest saving)")
        elif med <= -1.0:
            arm_results[t_arm] = ("INCONCLUSIVE", f"median {med:.2f}% but p={tt['wilcoxon_p']:.4f}")
            print(f"- {t_arm}: INCONCLUSIVE — median gap {med:.2f}% <= -1% but p >= alpha "
                  "(power question; NOT a falsification)")
        else:
            arm_results[t_arm] = ("NO-SUPPORT", f"median {med:.2f}%")
            print(f"- {t_arm}: NO-SUPPORT — median gap {med:.2f}% > -1%: not superior to "
                  "the strongest lossless ROI baseline")
        if len(present_thesis_arms) > 1:
            print("  [multiplicity note: co-primary arms at unadjusted alpha — "
                  "registered TODO: Holm or hierarchical procedure]")

    statuses = [v[0] for v in arm_results.values()]
    if "PASS" in statuses:
        winners = [a for a, (st, _) in arm_results.items() if st == "PASS"]
        verdict = f"GO — {', '.join(winners)} passed both gates; proceed to E2 additivity"
    elif "INCOMPLETE" in statuses:
        verdict = ("INCOMPLETE — required evidence missing (no A1, n below minimum, NaN "
                   "curves, or missing ceilings). No scientific conclusion; do NOT publish F1")
    elif "INCONCLUSIVE" in statuses:
        verdict = ("INCONCLUSIVE — valid comparisons ran; at least one arm points below the "
                   "-1% margin without clearing alpha. More sequences/power needed")
    elif "F2-FAIL" in statuses:
        verdict = ("F2-FAIL — G-1 passed but the information-loss gate did not; the honest "
                   "saving is bounded by the ceiling drop, publish as such")
    else:
        verdict = ("NO-SUPPORT — every co-primary arm was fully evaluated and none beat "
                   "the strongest lossless ROI baseline; scoped F1, publish E1 as null "
                   "+ ROI baseline")

    print(f"\n**VERDICT: {verdict}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
