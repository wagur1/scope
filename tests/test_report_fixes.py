"""Regression tests mirroring the external reviewer's fixtures (docs/REVIEW-EXTERNAL.md).

Each test reproduces one probe the reviewer ran against commit 032d3fa and
asserts the FIXED behaviour:
  P1-01  report merge with the runner's placeholder quality column
  P1-05  A4 ceiling inheritance from A2 (identical edited source)
  P2-01  duplicate / missing keys rejected (row counts alone are not enough)
  6.5    n<8 -> INCOMPLETE, not a falsification (and the p=1/32 correction)
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.report_e1 import ceiling_for, load  # noqa: E402
from scope_vcm.metrics import thesis_test  # noqa: E402


def _write_csvs(tmp_path: Path, e1_rows, q_rows, cel_rows, placeholder_quality=True):
    res = tmp_path / "results"
    res.mkdir(parents=True, exist_ok=True)
    e1_cols = ["seq", "arm", "qp", "delta", "rate_kbps"] + (
        ["quality", "is_ceiling", "notes"] if placeholder_quality else [])
    pd.DataFrame(e1_rows, columns=e1_cols).to_csv(res / "e1.csv", index=False)
    pd.DataFrame(q_rows, columns=["seq", "arm", "qp", "delta", "quality"]).to_csv(
        res / "quality.csv", index=False)
    pd.DataFrame(cel_rows, columns=["seq", "arm", "ceiling"]).to_csv(
        res / "ceiling.csv", index=False)
    return tmp_path


def _rows(placeholder_quality: bool = True):
    """One sequence, A0+A2, 3 QPs — placeholder quality matches eval quality."""
    e1, q, cel = [], [], []
    BQ = 0.5  # base quality value used for both ceilings
    for qp, rate in ((27, 100.0), (32, 60.0), (37, 30.0)):
        for arm in ("A0_anchor", "A2_preedit_fb"):
            r = rate if arm == "A0_anchor" else rate * 0.8
            row = ["s0", arm, qp, 0, round(r, 3)]
            if placeholder_quality:
                row += ["", 0, ""]
            e1.append(row)
            q.append(["s0", arm, qp, 0, BQ])
    cel.append(["s0", "A0_anchor", BQ])
    cel.append(["s0", "A2_preedit_fb", BQ])
    return e1, q, cel


def test_p1_01_merge_with_placeholder_quality_column(tmp_path):
    """Runner schema (placeholder quality in e1.csv) must merge cleanly —
    the reviewer reproduced KeyError: 'quality' on exactly this schema."""
    e1, q, cel = _rows(placeholder_quality=True)
    p = _write_csvs(tmp_path, e1, q, cel)
    df, cel_idx = load(p, primary_qps=None)
    assert "quality" in df.columns
    assert len(df) == 6
    assert not df.quality.isna().any()


def test_p2_01_duplicate_quality_key_rejected(tmp_path):
    """Row counts alone accepted dup+missing pairs (reviewer probe); now rejected."""
    e1, q, cel = _rows()
    q_bad = q + [q[0]]  # duplicate one key AND drop another — both must be caught
    q_bad = [r for i, r in enumerate(q_bad) if i != 3]
    p = _write_csvs(tmp_path, e1, q_bad, cel)
    with pytest.raises(ValueError):  # duplicate or missing — either rejection is correct
        load(p, primary_qps=None)


def test_p2_01_duplicate_ceiling_rejected(tmp_path):
    e1, q, cel = _rows()
    cel.append(["s0", "A0_anchor", 0.5])
    p = _write_csvs(tmp_path, e1, q, cel)
    with pytest.raises(ValueError, match="duplicate"):
        load(p, primary_qps=None)


def test_p1_05_a4_ceiling_inherits_a2(tmp_path):
    """A4 encodes A2's edited frames: missing A4 ceiling inherits A2's."""
    e1, q, cel = _rows()
    cel_idx = pd.DataFrame(cel, columns=["seq", "arm", "ceiling"]).set_index(["seq", "arm"])
    v, note = ceiling_for(cel_idx, "s0", "A4_preedit_fb_roi")
    assert v == pytest.approx(0.5)
    assert "inherited from A2" in note


def test_qp_primary_filter_slices_but_never_empties(tmp_path):
    e1, q, cel = _rows()
    p = _write_csvs(tmp_path, e1, q, cel)
    df, _ = load(p, primary_qps=[27])
    assert set(df.qp) == {27}
    with pytest.raises(ValueError, match="no rows left"):
        load(_write_csvs(tmp_path, e1, q, cel), primary_qps=[15])


def test_65_missing_n_is_incomplete_not_falsification():
    """n=5 (MOT17 reality) -> INCOMPLETE evidence; the registered minimum is a
    power rule, not a mathematical necessity (5 untied all-negative diffs give
    exact p = 1/32 = 0.03125 < 0.05)."""
    gaps = [-2.0] * 5
    tt = thesis_test(gaps)
    assert "reason" in tt and "incomplete" in tt["reason"].lower()
    # and the reviewer's arithmetic: exact one-sided tail at n=5 is 1/32
    from scipy import stats
    assert stats.wilcoxon(gaps, alternative="less").pvalue == pytest.approx(1 / 32)
