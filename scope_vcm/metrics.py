"""Decision statistics for the E1/E2 gates (MATH.md sections 6-9)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .bd import bd_rate

__all__ = ["thesis_test", "complementarity_kappa", "bootstrap_ci_median", "flicker_naive"]


def thesis_test(gaps: "np.ndarray", alpha: float = 0.05) -> dict:
    """Gate G-1 (MATH.md section 6): per-sequence BD gaps (proposal - ROI baseline).

    gaps: BD-rate(anchor, pre-edit) - BD-rate(anchor, ROI delta-QP), per sequence,
    in percent. Survives iff median <= -1.0 AND one-sided Wilcoxon p < alpha.
    NaN gaps HARD-FAIL (code review C1: dropping them is a data-dependent
    selection rule; the sequences with disjoint normalized intervals are the
    hard ones and must be reported, not filtered).
    """
    gaps = np.asarray(gaps, dtype=float)
    if np.any(~np.isfinite(gaps)):
        n_bad = int((~np.isfinite(gaps)).sum())
        return {"pass": False,
                "reason": f"{n_bad}/{len(gaps)} sequences have NaN BD "
                          "(disjoint normalized-quality intervals) — hard fail, no silent drop"}
    if len(gaps) < 8:
        # NOTE (external review 6.5 + Argon): the n>=8 rule is a REGISTERED
        # protocol minimum, NOT a mathematical necessity — 5 untied
        # all-negative differences already give exact one-sided p = 1/32 =
        # 0.03125 < 0.05. The rule exists for power/precision against the
        # 1-point margin, and missing n makes the run INCOMPLETE, not F1.
        return {"pass": False,
                "reason": f"registered minimum n=8 sequences, got {len(gaps)} "
                          "(incomplete evidence, not a falsification)"}
    med = float(np.median(gaps))
    try:
        stat, p = stats.wilcoxon(gaps, alternative="less")
    except ValueError:  # all-zero differences
        stat, p = float("nan"), 1.0
    # robustness: sign test (code review M3 — Wilcoxon symmetry is shaky for
    # ratio-derived BD gaps). Testing H1: P(gap < 0) > 0.5 -> alternative="greater" on n_neg.
    n_neg = int((gaps < 0).sum())
    p_sign = float(stats.binomtest(n_neg, len(gaps), 0.5, alternative="greater").pvalue)
    return {
        "pass": bool(med <= -1.0 and p < alpha),
        "median_gap_pct": med,
        "wilcoxon_p": float(p),
        "sign_test_p": p_sign,
        "n": int(len(gaps)),
        "alpha": alpha,
    }


def complementarity_kappa(anchor: "np.ndarray", x: "np.ndarray", y: "np.ndarray",
                          xy: "np.ndarray") -> float:
    """kappa (MATH.md section 7): super-additivity index, positive = super-additive.

    kappa = |BD(anchor, X+Y)| - |BD(anchor, X)| - |BD(anchor, Y)|.
    With BD<0 = saving, kappa > 0 means the combination saves MORE rate than
    the sum of the individual savings (complementary); kappa < 0 means the
    tools overlap (sub-additive/redundant).
    """
    return float(abs(bd_rate(anchor, xy)) - abs(bd_rate(anchor, x)) - abs(bd_rate(anchor, y)))


def bootstrap_ci_median(values: "np.ndarray", n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI (over sequences) for the median. Seed fixed for reproducibility."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(vals, size=len(vals), replace=True))
                     for _ in range(n_boot)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def flicker_naive(masks: "np.ndarray") -> float:
    """Frame-diff proxy of edit-map flicker (MATH.md section 8, no-flow fallback).

    d(A_t, A_{t-1}) = ||A_t - A_{t-1}||_1 / HW, averaged. NOTE: this is the
    no-optical-flow approximation; the warp-based metric needs flow fields
    and is E2t work.
    """
    a = np.asarray(masks, dtype=np.float32)
    if len(a) < 2:
        return 0.0
    diffs = [np.abs(a[t] - a[t - 1]).mean() for t in range(1, len(a))]
    return float(np.mean(diffs))
