"""Sanity tests for the BD implementation (docs/MATH.md section 5).

Run: python -m pytest tests/ -q   (or python tests/test_bd.py)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm.bd import _clean, bd_quality, bd_rate, ceiling_drop, ceiling_normalize  # noqa: E402


def _curve(q_range=(0.5, 0.9), rate_at=lambda q: 1e5 * (1.1 ** (q * 40))):
    qs = np.linspace(*q_range, 6)
    return np.column_stack([qs, rate_at(qs)])


def test_identical_curves_zero_bd():
    a = _curve()
    assert abs(bd_rate(a, a.copy())) < 1e-6
    assert abs(bd_quality(a, a.copy())) < 1e-6


def test_double_rate_is_plus_100():
    a = _curve()
    half = a.copy()
    half[:, 1] /= 2.0  # proposal reaches same quality at half the rate
    v = bd_rate(a, half)
    assert -52.0 < v < -48.0, f"expected ~-50% BD-rate, got {v}"


def test_quality_gain_negative_bd_rate():
    a = _curve()
    better = a.copy()
    better[:, 0] += 0.02  # proposal: same rate, higher quality
    v = bd_rate(a, better)
    assert v < -5.0, f"higher quality at same rate must save rate, got {v}"


def test_bd_quality_constant_shift():
    a = _curve()
    better = a.copy()
    better[:, 0] += 0.02  # constant quality offset at every rate
    v = bd_quality(a, better)
    assert abs(v - 0.02) < 5e-3, f"expected ~+0.02 mean quality gain, got {v}"


def test_disjoint_intervals_nan():
    a = _curve((0.1, 0.4))
    p = _curve((0.6, 0.9))
    assert np.isnan(bd_rate(a, p)), "no overlap -> NaN, never extrapolate"


def test_dedupe_averages_ties():
    pts = np.array([[0.5, 100.0], [0.5, 200.0], [0.7, 50.0]])
    out = _clean(pts)
    assert len(out) == 2
    assert out[0, 1] == pytest.approx(150.0)


def test_dedupe_duplicate_rates_for_quality_fit():
    # duplicate rates would make the bd_quality PCHIP x non-strictly-increasing
    pts = np.array([[0.5, 100.0], [0.7, 100.0], [0.9, 50.0]])
    out = _clean(pts)
    assert len(out) == 2  # rates deduped (mean quality), no exception
    assert bd_quality(pts, pts.copy()) == pytest.approx(0.0, abs=1e-6)


def test_dedupe_then_ratesort_keeps_quality_increasing():
    # self-review regression: rate-dedupe reordered rows by rate, silently
    # yielding NaN BD in bd_rate on legitimate dup-rate data
    pts = np.array([[0.5, 100.0], [0.7, 100.0], [0.9, 50.0]])
    out = _clean(pts)
    assert np.all(np.diff(out[:, 0]) > 0), f"quality axis must be increasing, got {out.tolist()}"
    assert np.isfinite(bd_rate(pts, pts.copy()))


def test_zero_rate_rejected():
    pts = np.array([[0.5, 0.0], [0.7, 50.0]])
    with pytest.raises(ValueError):
        _clean(pts)


def test_ceiling_normalize_and_drop():
    pts = np.array([[40.0, 100.0], [50.0, 200.0]])
    n = ceiling_normalize(pts, ceiling=50.0)
    assert n[1, 0] == pytest.approx(1.0)
    assert ceiling_drop(49.5, 50.0) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        ceiling_normalize(pts, ceiling=0.0)


def test_ceiling_drop_gate_boundary():
    # exactly 1 mAP point on a 50-point ceiling scale -> eta 0.02 -> FLAG
    assert ceiling_drop(49.0, 50.0) > 0.01


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
