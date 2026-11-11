"""Offline encoder/runner wiring checks; not a pretrained-model validation."""
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scope_vcm import codec, experiment
from scope_vcm.masks import save_masks


def _frames(path, hw=(18, 34), n=3):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for t in range(n):
        Image.fromarray(rng.integers(0, 256, (*hw, 3), dtype=np.uint8)).save(
            path / f"{t:04d}.png")
    return path


def test_roi_maps_use_macroblocks_and_keep_ctu_default(tmp_path):
    mask = np.zeros((18, 34), dtype=np.float32)
    mask[:16, :16] = 1
    mask[16:, 32:] = 1
    mb = codec.write_qp_delta_map([mask], 10, tmp_path / "mb.txt",
                                 mask.shape, block_size=16)
    assert mb.read_text().split() == ["0", "10", "10", "10", "10", "0"]
    ctu = codec.write_qp_delta_map([mask], 10, tmp_path / "ctu.txt", mask.shape)
    assert ctu.read_text().split() == ["10"]


def test_roi_mask_shape_and_finiteness_are_checked(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        codec.write_qp_delta_map([np.zeros((4, 4))], 10, tmp_path / "bad.txt", (18, 34))
    with pytest.raises(ValueError, match="non-finite"):
        codec.write_qp_delta_map([np.full((18, 34), np.nan)], 10,
                                 tmp_path / "bad.txt", (18, 34))


@pytest.mark.parametrize("text", ["0 0\n" * 3, "0 " * 18, "nan 0 0 0 0 0\n" * 3,
                                  "0 0 0 0 0 0\n" * 4, "x 0 0 0 0 0\n" * 3])
def test_roi_map_rejects_bad_shape_and_values(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        codec._validate_roi_map(path, 3, (18, 34))


def test_zero_map_and_i420_use_actual_dimensions(tmp_path):
    frames = _frames(tmp_path / "frames")
    out = codec._zero_map(tmp_path / "zero.txt", 3, (18, 34))
    assert len(out.read_text().splitlines()) == 3
    assert all(line.split() == ["0"] * 6 for line in out.read_text().splitlines())
    raw = tmp_path / "input.yuv"
    codec._pack_i420(frames, raw, 3, (18, 34))
    assert raw.stat().st_size == 3 * 18 * 34 * 3 // 2
    with pytest.raises(ValueError, match="frame size"):
        codec._pack_i420(frames, raw, 3, (18, 36))
    with pytest.raises(RuntimeError, match="pngs"):
        codec._pack_i420(frames, raw, 2, (18, 34))


def test_roi_encode_passes_dimensions_map_and_frame_count(tmp_path, monkeypatch):
    frames = _frames(tmp_path / "frames")
    tool = tmp_path / "tool.exe"
    tool.write_bytes(b"test double")
    cfg = codec.EncoderCfg(codec="x264roi", extra_params={"tool_path": str(tool)})
    commands = []

    def run(cmd, log):
        commands.append(cmd)
        if cmd[0] == str(tool):
            assert cmd[2:5] == ["34", "18", "24.0"]
            assert cmd[-1] == "3"
            codec._validate_roi_map(Path(cmd[6]), 3, (18, 34))
            Path(cmd[8]).write_bytes(b"encoded-double")
        else:
            decoded = Path(cmd[-1]).parent
            for fp in frames.glob("*.png"):
                shutil.copy2(fp, decoded / fp.name)

    monkeypatch.setattr(codec.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(codec, "_run", run)
    result = codec.encode(cfg, frames, tmp_path / "out", 3, fps=24.0, frame_size=(18, 34))
    assert result.bitstream.suffix == ".h264"
    assert result.n_frames == 3
    assert result.bitrate_kbps > 0
    assert len(commands) == 2


@pytest.mark.parametrize("hw", [None, (17, 34), (18, 33), (0, 34), (18.0, 34)])
def test_roi_encode_requires_valid_dimensions(tmp_path, monkeypatch, hw):
    tool = tmp_path / "tool.exe"
    tool.write_bytes(b"test double")
    monkeypatch.setattr(codec.shutil, "which", lambda name: "ffmpeg")
    cfg = codec.EncoderCfg(codec="x264roi", extra_params={"tool_path": str(tool)})
    with pytest.raises(ValueError):
        codec.encode(cfg, tmp_path, tmp_path / "out", 3, frame_size=hw)


def test_roi_encode_requires_map_for_intervention(tmp_path, monkeypatch):
    tool = tmp_path / "tool.exe"
    tool.write_bytes(b"test double")
    monkeypatch.setattr(codec.shutil, "which", lambda name: "ffmpeg")
    cfg = codec.EncoderCfg(codec="x264roi", roi_delta=10,
                           extra_params={"tool_path": str(tool)})
    with pytest.raises(ValueError, match="requires a map"):
        codec.encode(cfg, tmp_path, tmp_path / "out", 3, frame_size=(18, 34))


def test_offline_e1_all_arms_resume_and_symmetric_ceilings(tmp_path, monkeypatch):
    frames = _frames(tmp_path / "frames")
    mask = np.zeros((18, 34), dtype=np.float32)
    mask[:16, :16] = 1
    save_masks(tmp_path / "masks" / "tiny.npz", {t: mask for t in range(3)})
    config = {
        "workdir": str(tmp_path / "work"), "encoder": "x264roi", "tool_path": "double",
        "fps": 24.0, "preset": None, "qp_grid": [27, 32], "roi_deltas": [10],
        "masks_dir": str(tmp_path / "masks"), "gt_json": str(tmp_path / "gt.json"),
        "arms": {"A0_anchor": True, "A1_roi": True, "A2_preedit_fb": True,
                 "A2c_preedit_blurall": True, "A3_preedit_scope": True,
                 "A4_preedit_fb_roi": False},
        "scope_repo": "double", "scope_ckpt": "double",
        "sequences": [{"id": "tiny", "frames": str(frames)}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))
    calls = []

    def encode(cfg, src, out, n_frames, fps, frame_size=None):
        calls.append(cfg)
        if not cfg.lossless:
            assert cfg.codec == "x264roi"
            assert frame_size == (18, 34)
        else:
            assert cfg.preset == "medium"
            assert frame_size == (18, 34)
        if cfg.roi_map:
            codec._validate_roi_map(cfg.roi_map, 3, (18, 34))
            assert cfg.roi_map.read_text().splitlines()[0].split() == ["0", "10", "10", "10", "10", "10"]
        decoded = out / "decoded"
        decoded.mkdir(parents=True)
        for fp in src.glob("*.png"):
            shutil.copy2(fp, decoded / fp.name)
        bitstream = out / "enc.h264"
        bitstream.write_bytes(b"offline-double")
        return codec.EncodeResult(100.0, bitstream, decoded, n_frames)

    class EditorDouble:
        def begin_sequence(self, sid):
            pass

        def edit_frame(self, img, t):
            return img

    original_build_editor = experiment.build_editor
    monkeypatch.setattr(experiment, "build_editor", lambda cfg, **kwargs:
                        EditorDouble() if cfg["type"] == "scope" else original_build_editor(cfg, **kwargs))
    monkeypatch.setattr(experiment, "encode", encode)
    csv_path = experiment.run_e1(config_path)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 10
    assert {r["arm"] for r in rows} == {a for a, enabled in config["arms"].items() if enabled}
    experiment.run_e1(config_path)
    assert len(calls) == 10
    assert len(list(csv.DictReader(csv_path.open()))) == 10

    spec = importlib.util.spec_from_file_location("eval_e1_integration", Path(__file__).resolve().parent.parent / "scripts" / "eval_e1.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    monkeypatch.setattr(evaluation, "encode", encode)

    def detect(path, weights, classes=None):
        assert len(list(path.glob("*.png"))) == 3
        return []

    monkeypatch.setattr(evaluation, "detect_frames", detect)
    monkeypatch.setattr(evaluation, "mAP", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(sys, "argv", ["eval_e1.py", str(config_path), "--weights", "offline-double"])
    assert evaluation.main() == 0
    ceilings = list(csv.DictReader((csv_path.parent / "ceiling.csv").open()))
    qualities = list(csv.DictReader((csv_path.parent / "quality.csv").open()))
    assert len(ceilings) == 4
    assert len(qualities) == 10
    assert all(c.codec == "x264" and c.lossless for c in calls[10:])
    assert evaluation.main() == 0
    assert len(calls) == 14


def test_real_ffmpeg_symmetric_lossless_roundtrip(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    frames = _frames(tmp_path / "frames")
    decoded_outputs = []
    for name in ("original", "edited"):
        result = codec.encode(codec.EncoderCfg(codec="x264", lossless=True, preset="veryfast"),
                              frames, tmp_path / name, 3, fps=24.0)
        assert result.n_frames == 3
        decoded_outputs.append(result.decoded_dir)
        for fp in frames.glob("*.png"):
            with Image.open(fp) as expected, Image.open(result.decoded_dir / fp.name) as actual:
                # Lossless coding is in YUV444; RGB conversion still rounds.
                error = np.abs(np.asarray(actual).astype(int) - np.asarray(expected).astype(int))
                assert error.max() <= 2
    for fp in frames.glob("*.png"):
        with Image.open(decoded_outputs[0] / fp.name) as a, Image.open(decoded_outputs[1] / fp.name) as b:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_decoded_frames_must_be_zero_indexed_and_correct_size(tmp_path):
    decoded = _frames(tmp_path / "decoded")
    codec._check_decoded(decoded, 3, (18, 34))
    with pytest.raises(RuntimeError, match="decoded size"):
        codec._check_decoded(decoded, 3, (18, 36))
    (decoded / "0000.png").rename(decoded / "0003.png")
    with pytest.raises(RuntimeError, match="zero-indexed"):
        codec._check_decoded(decoded, 3, (18, 34))


def test_encoder_failures_stop_decode_and_keep_logs(tmp_path, monkeypatch):
    frames = _frames(tmp_path / "frames")
    tool = tmp_path / "tool.exe"
    tool.write_bytes(b"test double")
    cfg = codec.EncoderCfg(codec="x264roi", extra_params={"tool_path": str(tool)})
    commands = []

    def failing_run(cmd, **kwargs):
        from types import SimpleNamespace
        commands.append(cmd)
        return SimpleNamespace(returncode=2, stdout="", stderr="bad ROI input")

    monkeypatch.setattr(codec.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(codec.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="command failed"):
        codec.encode(cfg, frames, tmp_path / "out", 3, frame_size=(18, 34))
    assert len(commands) == 1
    assert "bad ROI input" in (tmp_path / "out" / "encode.log").read_text()
