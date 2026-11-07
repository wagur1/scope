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


def complementarity_kappa_multiplicative(anchor: "np.ndarray", x: "np.ndarray",
                                         y: "np.ndarray", xy: "np.ndarray",
                                         grid: int = 512) -> float:
    """Multiplicative interaction (review Astra: the |BD|-difference kappa
    miscounts independent multiplicative savings — two 20% cuts combine to
    -36%, i.e. -4pp under the old definition — and counts regressions as
    savings).

    On the common normalized-quality interval: kappa_log = mean over q of
    [log2 R_xy(q) - log2 R_x(q) - log2 R_y(q) + log2 R_a(q)]; returns
    1 - 2^kappa_log = the rate saving of X+Y BEYOND the multiplicative
    combination of the individual savings. 0 = exactly multiplicative
    (independent tools); POSITIVE = super-additive (synergy — the pair
    saves more than the product of individual savings); negative =
    sub-additive overlap. (Stale-bytecode note: verify with the
    independent-tools sanity case before trusting any kappa number.)
    """
    from scipy.interpolate import PchipInterpolator

    def fit(pts):
        pts = np.asarray(pts, dtype=float)
        pts = pts[np.argsort(pts[:, 0], kind="stable")]
        return PchipInterpolator(pts[:, 0], np.log2(pts[:, 1]), extrapolate=False)

    a = np.asarray(anchor, dtype=float)
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    xyy = np.asarray(xy, dtype=float)
    q_lo = max(a[:, 0].min(), xx[:, 0].min(), yy[:, 0].min(), xyy[:, 0].min())
    q_hi = min(a[:, 0].max(), xx[:, 0].max(), yy[:, 0].max(), xyy[:, 0].max())
    if not q_lo < q_hi:
        return float("nan")
    g = np.linspace(q_lo, q_hi, grid)
    fa, fx, fy, fxy = fit(a), fit(xx), fit(yy), fit(xyy)
    term = fxy(g) - fx(g) - fy(g) + fa(g)
    term = term[np.isfinite(term)]
    if len(term) < 8:
        return float("nan")
    fin = np.isfinite(term)
    if fin.sum() < 8:
        return float("nan")
    kappa_log = float(np.trapezoid(term[fin], g[fin]) / (g[fin][-1] - g[fin][0]))
    return float(1.0 - 2.0 ** kappa_log)


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
