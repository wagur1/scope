"""Regression tests for whole-edit attenuation and strength-aware reuse."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm import experiment, preedit
from scripts import eval_e1, report_e1


@pytest.mark.parametrize("strength", [-.1, 1.1, float("nan"), float("inf"), True, ".25", None])
def test_invalid_strength_rejected_before_editor_loading(strength, monkeypatch):
    monkeypatch.setattr(preedit, "ScopePreEditor", lambda *args: pytest.fail("loaded editor"))
    with pytest.raises(ValueError, match="edit_strength"):
        preedit.build_editor({"type": "scope", "edit_strength": strength})


def test_zero_is_exact_identity_without_masks_or_checkpoint(monkeypatch):
    monkeypatch.setattr(preedit, "ScopePreEditor", lambda *args: pytest.fail("loaded editor"))
    img = np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 47
    for kind in ("scope", "blurbg"):
        editor = preedit.build_editor({"type": kind, "edit_strength": 0})
        editor.begin_sequence("s0")
        result = editor.edit_frame(img, 0)
        np.testing.assert_array_equal(result, img)
        assert result is not img and result.dtype == np.float32


@pytest.mark.parametrize("kind", ["scope", "blurbg"])
def test_full_strength_compatibility_and_single_total_blend(kind, monkeypatch):
    class Editor:
        name = "fake_scope"

        def __init__(self, *args):
            pass

        def begin_sequence(self, sid):
            pass

        def edit_frame(self, image, t):
            return 1 - image

    monkeypatch.setattr(preedit, "ScopePreEditor", Editor)
    cfg = {"type": kind, "scope_repo": "unused", "ckpt_path": "unused", "sigma": 1.}
    masks = lambda sid, t: np.zeros((4, 4), np.float32)
    img = np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 47
    original = img.copy()
    outputs = []
    for extra in ({}, {"edit_strength": 1}, {"edit_strength": .25}):
        editor = preedit.build_editor(dict(cfg, **extra), masks)
        editor.begin_sequence("s0")
        outputs.append(editor.edit_frame(img, 0))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_allclose(outputs[2], img + .25 * (outputs[0] - img), atol=1e-7)
    np.testing.assert_array_equal(img, original)
    assert outputs[2].dtype == np.float32 and outputs[2].shape == img.shape


def setup_run(tmp_path, monkeypatch, strength=0):
    frames = tmp_path / "frames"
    frames.mkdir()
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5
    for t in range(2):
        Image.fromarray(image).save(frames / f"{t:04d}.png")
    root = tmp_path / "work"
    config = {"workdir": str(root), "encoder": "x264", "qp_grid": [22], "roi_deltas": [4],
              "edit_strength": strength, "blur_sigma": 1., "gt_json": "unused.json",
              "arms": {a: True for a in ("A0_anchor", "A1_roi", "A2_preedit_fb", "A2c_preedit_blurall",
                                        "A3_preedit_scope", "A4_preedit_fb_roi")},
              "sequences": [{"id": "s0", "frames": str(frames)}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    class Masks:
        def __init__(self, *args):
            pass

        def __getitem__(self, t):
            return np.zeros((4, 4), np.float32)

    calls = []

    def encode(cfg, source, out, **kwargs):
        calls.append((out.parent.parent.name, Path(source), experiment._load_frame(Path(source), 0)))
        return SimpleNamespace(bitrate_kbps=100., decoded_dir=Path(source))

    monkeypatch.setattr(experiment, "LazyMasks", Masks)
    monkeypatch.setattr(experiment, "encode", encode)
    monkeypatch.setattr(experiment, "write_qp_delta_map", lambda masks, d, path, *args, **kwargs: path)
    return config, path, root, frames, calls


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_runner_zero_all_arms_and_resume_identity(tmp_path, monkeypatch):
    cfg, path, root, frames, calls = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(preedit, "ScopePreEditor", lambda *args: pytest.fail("loaded editor"))
    experiment.run_e1(path)
    assert len(calls) == 6
    original = experiment._load_frame(frames, 0)
    for _, _, output in calls:
        np.testing.assert_array_equal(output, original)
    assert calls[0][1] == calls[1][1] == frames
    assert calls[-1][1] == calls[2][1]  # A4 uses A2 once.
    identity = json.loads((root / "results" / "run_identity.json").read_text())
    assert identity["edit_strength"] == 0.
    manifest = json.loads(next((root / "results" / "manifests").glob("*.json")).read_text())
    assert manifest["edit_strength"] == 0. and manifest["run_identity"] == identity
    experiment.run_e1(path)
    assert len(calls) == 6
    before = snapshot(root)
    cfg["edit_strength"] = .25
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="fresh workdir"):
        experiment.run_e1(path)
    assert snapshot(root) == before


def test_runner_fractional_blend_and_anchor_routing(tmp_path, monkeypatch):
    cfg, path, root, frames, calls = setup_run(tmp_path, monkeypatch, .25)

    class Editor:
        name = "fake"

        def __init__(self, *args):
            pass

        def begin_sequence(self, sid):
            pass

        def edit_frame(self, img, t):
            return 1 - img

    monkeypatch.setattr(preedit, "ScopePreEditor", Editor)
    cfg.update(scope_repo="unused", scope_ckpt="unused")
    path.write_text(yaml.safe_dump(cfg))
    experiment.run_e1(path)
    outputs = {arm: image for arm, _, image in calls}
    original = experiment._load_frame(frames, 0)
    full_blur = preedit._gaussian_blur(original, 1.)
    quantize = lambda image: (np.clip(image, 0, 1) * 255 + .5).astype(np.uint8).astype(np.float32) / 255
    np.testing.assert_array_equal(outputs["A0_anchor"], original)
    np.testing.assert_array_equal(outputs["A1_roi"], original)
    for arm in ("A2_preedit_fb", "A2c_preedit_blurall", "A4_preedit_fb_roi"):
        np.testing.assert_array_equal(outputs[arm], quantize(original + .25 * (full_blur - original)))
    np.testing.assert_array_equal(outputs["A3_preedit_scope"], quantize(original + .25 * (1 - 2 * original)))


@pytest.mark.parametrize("artifact", ["edited/A2/s0/0000.png", "enc/A0/s0/bitstream", "results/e1.csv", "results/quality.csv"])
def test_legacy_artifacts_refused_without_modification(tmp_path, monkeypatch, artifact):
    cfg, path, root, _, _ = setup_run(tmp_path, monkeypatch)
    old = root / artifact
    old.parent.mkdir(parents=True)
    old.write_bytes(b"legacy")
    before = snapshot(root)
    with pytest.raises(ValueError, match="fresh workdir"):
        experiment.run_e1(path)
    assert snapshot(root) == before


def test_invalid_strength_has_no_runner_side_effects(tmp_path, monkeypatch):
    cfg, path, root, _, _ = setup_run(tmp_path, monkeypatch)
    cfg["edit_strength"] = float("nan")
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="edit_strength"):
        experiment.run_e1(path)
    assert not root.exists()


def test_eval_report_use_saved_strength_and_refuse_mismatch(tmp_path, monkeypatch, capsys):
    cfg, path, root, _, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(preedit, "ScopePreEditor", lambda *args: pytest.fail("loaded editor"))
    experiment.run_e1(path)
    monkeypatch.setattr(eval_e1, "encode", lambda ecfg, source, out, **kwargs: SimpleNamespace(decoded_dir=Path(source)))
    monkeypatch.setattr(eval_e1, "detect_frames", lambda *args, **kwargs: [])
    monkeypatch.setattr(eval_e1, "mAP", lambda *args, **kwargs: .5)
    monkeypatch.setattr(sys, "argv", ["eval", str(path), "--weights", "unused"])
    assert eval_e1.main() == 0
    monkeypatch.setattr(sys, "argv", ["report", str(root)])
    assert report_e1.main() == 0
    assert "edit_strength=0" in capsys.readouterr().out
    before = snapshot(root)
    cfg["edit_strength"] = .25
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["eval", str(path), "--weights", "unused"])
    with pytest.raises(ValueError, match="fresh workdir"):
        eval_e1.main()
    monkeypatch.setattr(sys, "argv", ["report", str(root), "--config", str(path)])
    with pytest.raises(ValueError, match="fresh workdir"):
        report_e1.main()
    assert snapshot(root) == before


def test_omitted_and_explicit_full_strength_share_identity(tmp_path, monkeypatch):
    cfg, path, root, _, _ = setup_run(tmp_path, monkeypatch)
    cfg.pop("edit_strength")
    cfg["arms"]["A3_preedit_scope"] = False
    path.write_text(yaml.safe_dump(cfg))
    experiment.run_e1(path)
    cfg["edit_strength"] = 1
    path.write_text(yaml.safe_dump(cfg))
    experiment.run_e1(path)
    assert json.loads((root / "results" / "run_identity.json").read_text())["edit_strength"] == 1.
