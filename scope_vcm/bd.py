"""Bjontegaard delta computation with ceiling normalization.

Implements the exact algorithm specified in docs/MATH.md section 5:
  1. sort + deduplicate quality points (mean rate at ties)
  2. common quality interval (no extrapolation -> NaN when empty)
  3. monotone PCHIP fit of log2(rate) as a function of quality
  4. BD-rate = (2^(mean log2-rate difference over the interval) - 1) * 100%

Sign convention: bd_rate(anchor, proposal) < 0 means the proposal saves rate.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = [
    "dedupe_quality",
    "bd_rate",
    "bd_rate_truncated",
    "bd_quality",
    "ceiling_normalize",
    "ceiling_drop",
]


def _clean(points: "np.ndarray") -> "np.ndarray":
    """Sort by quality; collapse duplicate qualities (mean rate) AND duplicate
    rates (mean quality) — PCHIP requires strictly increasing x on BOTH fits
    (bd_rate fits on quality, bd_quality fits on log2-rate).

    points: (n, 2) array of [quality, rate].
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be shape (n, 2) = [quality, rate]")
    if len(pts) < 2:
        raise ValueError("need at least 2 RD points")
    if not np.all(np.isfinite(pts)):
        raise ValueError("non-finite RD points")
    if np.any(pts[:, 1] <= 0):
        raise ValueError("rates must be > 0 for log2 fit")
    pts = pts[np.argsort(pts[:, 0], kind="stable")]

    def _dedupe(vals: "np.ndarray", other: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        u, inverse = np.unique(vals, return_inverse=True)
        if len(u) == len(vals):
            return vals, other
        agg = np.zeros_like(u)
        for k in range(len(u)):
            agg[k] = other[inverse == k].mean()
        return u, agg

    q, r = _dedupe(pts[:, 0], pts[:, 1])
    r, q = _dedupe(r, q)  # duplicate rates break the bd_quality fit
    # rate-dedupe reorders rows by rate; re-sort by quality so bd_rate's PCHIP
    # x-axis (quality) stays strictly increasing (self-review bug fix 1)
    order = np.argsort(q, kind="stable")
    return np.column_stack([q[order], r[order]])


def bd_rate(anchor: "np.ndarray", proposal: "np.ndarray") -> float:
    """BD-rate in percent (negative = proposal saves rate vs anchor).

    anchor, proposal: (n, 2) arrays of [quality, rate]. Quality axis should be
    ceiling-normalized (see `ceiling_normalize`) when arms edit content.
    """
    a = _clean(anchor)
    p = _clean(proposal)
    q_lo = max(a[0, 0], p[0, 0])
    q_hi = min(a[-1, 0], p[-1, 0])
    if not q_lo < q_hi:
        return float("nan")  # no common quality interval: BD undefined

    qa, ra = a[:, 0], a[:, 1]
    qp, rp = p[:, 0], p[:, 1]
    f_a = PchipInterpolator(qa, np.log2(ra), extrapolate=False)
    f_p = PchipInterpolator(qp, np.log2(rp), extrapolate=False)

    grid = np.linspace(q_lo, q_hi, 512)
    diff = f_p(grid) - f_a(grid)
    finite = np.isfinite(diff)
    if finite.sum() < 8:
        return float("nan")
    q_lo, q_hi = grid[finite][0], grid[finite][-1]
    delta = float(np.trapezoid(diff[finite], grid[finite]) / (q_hi - q_lo))
    return (2.0 ** delta - 1.0) * 100.0


def bd_rate_truncated(anchor: "np.ndarray", proposal: "np.ndarray", q_cap: float) -> float:
    """Secondary BD-rate on the common absolute quality interval truncated at
    q_cap = min(m*_E, m*_anchor) (MATH.md section 4, deployment estimand).
    """
    if not np.isfinite(q_cap):
        return float("nan")
    a = _clean(anchor)
    p = _clean(proposal)
    q_lo = max(a[0, 0], p[0, 0])
    q_hi = min(a[-1, 0], p[-1, 0], float(q_cap))
    if not q_lo < q_hi:
        return float("nan")

    qa, ra = a[:, 0], a[:, 1]
    qp, rp = p[:, 0], p[:, 1]
    f_a = PchipInterpolator(qa, np.log2(ra), extrapolate=False)
    f_p = PchipInterpolator(qp, np.log2(rp), extrapolate=False)

    grid = np.linspace(q_lo, q_hi, 512)
    diff = f_p(grid) - f_a(grid)
    finite = np.isfinite(diff)
    if finite.sum() < 8:
        return float("nan")
    q_lo, q_hi = grid[finite][0], grid[finite][-1]
    delta = float(np.trapezoid(diff[finite], grid[finite]) / (q_hi - q_lo))
    return (2.0 ** delta - 1.0) * 100.0


def bd_quality(anchor: "np.ndarray", proposal: "np.ndarray") -> float:
    """BD-quality: mean quality gain of the proposal over the common log2-rate interval.

    Mirror of bd_rate: PCHIP fit of quality as a function of log2(rate),
    integrated over the common log2-rate interval (mean quality difference).
    """
    a = _clean(anchor)
    p = _clean(proposal)
    # bd_quality fits on log2(rate): need rows ordered by rate (self-review fix)
    a = a[np.argsort(a[:, 1], kind="stable")]
    p = p[np.argsort(p[:, 1], kind="stable")]
    lr_lo = max(np.log2(a[:, 1].min()), np.log2(p[:, 1].min()))
    lr_hi = min(np.log2(a[:, 1].max()), np.log2(p[:, 1].max()))
    if not lr_lo < lr_hi:
        return float("nan")

    g_a = PchipInterpolator(np.log2(a[:, 1]), a[:, 0], extrapolate=False)
    g_p = PchipInterpolator(np.log2(p[:, 1]), p[:, 0], extrapolate=False)

    grid = np.linspace(lr_lo, lr_hi, 512)
    diff = g_p(grid) - g_a(grid)
    finite = np.isfinite(diff)
    if finite.sum() < 8:
        return float("nan")
    lo, hi = grid[finite][0], grid[finite][-1]
    return float(np.trapezoid(diff[finite], grid[finite]) / (hi - lo))


def ceiling_normalize(points: "np.ndarray", ceiling: float) -> "np.ndarray":
    """Divide the quality axis by the uncompressed ceiling (MATH.md section 4).

    ceiling = mAP of the method's own output at negligible compression.
    Normalized quality is 1.0 at the ceiling by construction.
    """
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError(f"ceiling must be finite positive, got {ceiling}")
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] /= ceiling
    return pts


def ceiling_drop(edited_ceiling_mAP: float, anchor_ceiling_mAP: float) -> float:
    """eta = 1 - m*_E / m*_anchor (MATH.md section 4). Gate: eta <= 0.01."""
    if anchor_ceiling_mAP <= 0:
        raise ValueError("anchor ceiling must be positive")
    return 1.0 - edited_ceiling_mAP / anchor_ceiling_mAP
