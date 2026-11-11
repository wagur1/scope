"""Offline kernel integration checks: no encoder, network or ML imports needed."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


KERNEL_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "real_e1_kernel.py"


def load_kernel():
    spec = importlib.util.spec_from_file_location("real_e1_kernel_under_test", KERNEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    module = load_kernel()
    monkeypatch.setattr(module, "WORK", tmp_path)
    monkeypatch.setattr(module, "REPO", tmp_path / "scope")
    monkeypatch.setattr(module, "record_gpu", lambda summary: summary.update(gpu="mocked"))
    return module


def write(path, data=b"asset"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def offline(kernel, monkeypatch):
    """Exercise the real run helper, main, assets, config and publisher with fake commands."""
    write(kernel.REPO / "tools" / "x264_roi_encode.c", b"C tool source")
    write(kernel.REPO / "user_source.py", b"keep user repo")
    write(kernel.WORK / "SCOPE" / "scope" / "inference.py", b"keep user inference")
    write(kernel.WORK / "SCOPE" / "runs" / "user" / "best_val.pt", b"keep user weights")
    dataset = kernel.WORK / "dataset"
    write(dataset / "annotations" / "train.json", b'{"images": []}')
    monkeypatch.setattr(kernel, "find_dataset_root", lambda: dataset)
    calls = []
    controls = SimpleNamespace(fail=None, no_tool=False, bad_source=False, no_weights=False,
                               calls=calls, prepared=False, scope_dir=None, scope_ckpt=None)

    def fake_subprocess(cmd, cwd=None, **kwargs):
        assert kwargs["capture_output"] and kwargs["text"]
        assert kwargs["timeout"] > 0
        cmd = [str(arg) for arg in cmd]
        if cmd[:2] == ["git", "clone"]:
            name = "scope_clone" if "miccunifi/SCOPE" in cmd[-2] else "clone"
        elif cmd[:2] == ["git", "rev-parse"]:
            name = "git_sha" if Path(cwd) == kernel.REPO else "scope_sha"
        elif cmd[0] == "apt-get":
            name = "build_deps"
        elif cmd[0] == "gcc":
            name = "build_tool"
        elif cmd[:3] == ["python", "-m", "gdown"]:
            name = "scope_weights"
        elif cmd[:3] == ["python", "-m", "pip"]:
            name = "scope_pip" if "gdown" in cmd else "pip"
        else:
            name = {"scripts/run_e1.py": "run_e1", "scripts/eval_e1.py": "eval_e1",
                    "scripts/report_e1.py": "report_e1"}[cmd[1]]
        calls.append((name, cmd))
        if controls.fail == name:
            return SimpleNamespace(returncode=9, stdout="partial output", stderr="mock failure")
        if name == "build_tool" and not controls.no_tool:
            write(Path(cmd[cmd.index("-o") + 1]), b"compiled tool")
        if name == "scope_clone":
            source = Path(cmd[-1]) / "scope" / "inference.py"
            if not controls.bad_source:
                write(source, b"def load_scope_model(): pass\ndef apply_scope_bgr(): pass\n")
            else:
                Path(cmd[-1]).mkdir(parents=True)
        if name == "scope_weights" and not controls.no_weights:
            runs = Path(cmd[-1])
            write(runs / "z_ablation" / "best_val.pt", b"other checkpoint")
            write(runs / "a_selected" / "best_val.pt", b"selected checkpoint")
            write(runs / "many_extra_weights.pt", b"extra")
        if name in {"run_e1", "eval_e1", "report_e1"}:
            assert (controls.scope_dir / "scope" / "inference.py").is_file()
            assert controls.scope_ckpt.is_file()
            assert not (controls.scope_dir / "runs").exists()
        stdout = "VERDICT: INCOMPLETE\n" if name == "report_e1" else "mock sha\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(kernel.subprocess, "run", fake_subprocess)

    def fake_prepare(seq, frame_limit, qps):
        controls.prepared = True
        scope_dir, scope_ckpt = kernel.prepare_scope_assets()
        controls.scope_dir, controls.scope_ckpt = scope_dir, scope_ckpt
        root = kernel.WORK / "real_e1"
        cfg = kernel.build_config(root, [{"id": seq, "frames": str(root / "frames")}],
                                  qps, scope_dir, scope_ckpt)
        config = write(root / "e1_real.yaml", json.dumps(cfg).encode())
        write(root / "work" / "enc" / "partial.bin", b"partial encoded data")
        for name in ("e1.csv", "quality.csv", "ceiling.csv"):
            write(root / "work" / "results" / name, b"seq,value\ns0,1\n")
        return cfg, config

    monkeypatch.setattr(kernel, "prepare", fake_prepare)
    return controls


def test_collection_import_needs_no_heavy_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"cv2", "torch", "numpy", "yaml", "ultralytics"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    load_kernel()


def test_config_arms_deltas_and_actual_encoder_policy(kernel):
    tool = write(kernel.WORK / "tools" / "x264_roi_encode")
    scope_dir = kernel.WORK / "owned_scope"
    write(scope_dir / "scope" / "inference.py")
    checkpoint = write(kernel.WORK / "selected.pt")
    cfg = kernel.build_config(kernel.WORK / "real_e1", [], [27, 32, 37], scope_dir, checkpoint)
    assert cfg["encoder"] == "x264roi"
    assert cfg["tool_path"] == str(tool)
    assert cfg["roi_deltas"] == [4, 8, 12, 16]
    assert cfg["qp_grid"] == [27, 32, 37]
    assert cfg["arms"] == {"A0_anchor": True, "A1_roi": True, "A2_preedit_fb": True,
                           "A2c_preedit_blurall": True, "A3_preedit_scope": True,
                           "A4_preedit_fb_roi": False}
    assert cfg["scope_repo"] == str(scope_dir)
    assert cfg["scope_ckpt"] == str(checkpoint)
    assert cfg["preset"] is None
    assert cfg["encoder_policy"]["rate_control"] == "CRF"
    assert cfg["encoder_policy"]["aq_mode"] == 1
    assert cfg["encoder_policy"]["aq_strength"] == 1.0
    assert cfg["encoder_policy"]["mbtree"] is False
    assert cfg["encoder_policy"]["named_preset"] is None


@pytest.mark.parametrize("missing", ["tool", "source", "checkpoint"])
def test_config_rejects_missing_required_assets(kernel, missing):
    scope_dir = kernel.WORK / "owned_scope"
    assets = {"tool": write(kernel.WORK / "tools" / "x264_roi_encode"),
              "source": write(scope_dir / "scope" / "inference.py"),
              "checkpoint": write(kernel.WORK / "selected.pt")}
    assets[missing].unlink()
    with pytest.raises(FileNotFoundError, match="Required experiment asset"):
        kernel.build_config(kernel.WORK / "real_e1", [], [27], scope_dir, assets["checkpoint"])


def test_scope_fetch_keeps_source_and_prunes_only_owned_weights(kernel, offline):
    scope_dir, checkpoint = kernel.prepare_scope_assets()
    assert (scope_dir / "scope" / "inference.py").is_file()
    assert checkpoint.read_bytes() == b"selected checkpoint"
    assert not (scope_dir / "runs").exists()
    assert (kernel.WORK / "SCOPE" / "scope" / "inference.py").read_bytes() == b"keep user inference"
    assert (kernel.WORK / "SCOPE" / "runs" / "user" / "best_val.pt").read_bytes() == b"keep user weights"
    assert kernel.summary["scope_ckpt_origin"] == "runs/a_selected/best_val.pt"
    assert kernel.summary["scope_ckpt_sha256"] == hashlib.sha256(b"selected checkpoint").hexdigest()
    assert kernel.summary["scope_source_sha256"] == kernel.sha256(scope_dir / "scope" / "inference.py")


@pytest.mark.parametrize("failed_stage", ["build_deps", "build_tool", "pip", "scope_clone",
                                         "scope_pip", "scope_weights", "run_e1", "eval_e1",
                                         "report_e1"])
def test_pipeline_fail_fast_and_publishes_failure(kernel, offline, failed_stage):
    offline.fail = failed_stage
    with pytest.raises(RuntimeError, match=failed_stage):
        kernel.main(shard=8, frame_limit=3)
    names = [name for name, _ in offline.calls]
    assert names[-1] == failed_stage
    if failed_stage in {"build_deps", "build_tool", "pip"}:
        assert not offline.prepared
    assert kernel.REPO.is_dir()
    assert (kernel.WORK / "SCOPE" / "runs" / "user" / "best_val.pt").is_file()
    if offline.scope_dir is not None:
        assert (offline.scope_dir / "scope" / "inference.py").is_file()
        assert (kernel.WORK / "real_e1" / "work" / "enc" / "partial.bin").is_file()
    assert (kernel.WORK / "logs" / f"{failed_stage}.log").is_file()
    for dest in (kernel.WORK, kernel.WORK / "out", kernel.WORK / "0000deliv"):
        summary = json.loads((dest / "summary.json").read_text())
        assert summary["status"] == "failed"
        assert summary["steps"][failed_stage]["returncode"] == 9
        assert summary["shard"]["replicate"] == 1
        assert "verdict" not in summary and "result_rows" not in summary
        assert "mock failure" in (dest / "failure.log").read_text()
        assert not (dest / "report.md").exists()
    assert not list((kernel.WORK / "out").glob("*.csv"))


@pytest.mark.parametrize("invalid", ["no_tool", "bad_source", "no_weights"])
def test_missing_build_or_scope_output_is_fatal(kernel, offline, invalid):
    setattr(offline, invalid, True)
    if invalid == "no_tool":
        # exists() alone would accept a directory as a built tool.
        (kernel.WORK / "tools" / "x264_roi_encode").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        kernel.main()
    names = [name for name, _ in offline.calls]
    assert "run_e1" not in names
    summary = json.loads((kernel.WORK / "0000deliv" / "summary.json").read_text())
    assert summary["status"] == "failed"
    if invalid == "no_tool":
        assert summary["tool_built"] is False
        assert "pip" not in names
    elif invalid == "bad_source":
        assert "scope_pip" not in names
    else:
        owned_source = Path(summary["scope_assets_dir"]) / "SCOPE" / "scope" / "inference.py"
        assert owned_source.is_file()


def test_timeout_has_log_and_failure_deliverables(kernel, offline, monkeypatch):
    def timeout(*args, **kwargs):
        raise kernel.subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(kernel.subprocess, "run", timeout)
    with pytest.raises(kernel.subprocess.TimeoutExpired):
        kernel.main()
    assert (kernel.WORK / "logs" / "git_sha.log").is_file()
    summary = json.loads((kernel.WORK / "0000deliv" / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["steps"]["git_sha"]["returncode"] is None
    assert (kernel.WORK / "0000deliv" / "failure.log").is_file()


@pytest.mark.parametrize("failed_stage", ["build_tool", "run_e1"])
def test_failure_after_success_archives_previous_deliverables(kernel, offline, failed_stage):
    kernel.main(frame_limit=3, qp_grid=[27])
    destinations = {"root": kernel.WORK, "out": kernel.WORK / "out",
                    "0000deliv": kernel.WORK / "0000deliv"}
    previous = {}
    for label, dest in destinations.items():
        names = ["summary.json", "report.md"]
        if label != "root":
            names += ["e1.csv", "quality.csv", "ceiling.csv"]
        previous[label] = {name: (dest / name).read_bytes() for name in names}
    # Unrelated assets must remain untouched even inside a publication directory.
    user_asset = write(kernel.WORK / "out" / "user.csv", b"historical user data")
    offline.fail = failed_stage
    with pytest.raises(RuntimeError, match=failed_stage):
        kernel.main(frame_limit=3, qp_grid=[27])
    for label, dest in destinations.items():
        current = json.loads((dest / "summary.json").read_text())
        assert current["status"] == "failed"
        assert "verdict" not in current and "result_rows" not in current
        assert (dest / "failure.log").is_file()
        assert not (dest / "report.md").exists()
        if label != "root":
            assert all(not (dest / name).exists() for name in ("e1.csv", "quality.csv", "ceiling.csv"))
        archive = Path(current["previous_deliverables"]) / label
        for name, data in previous[label].items():
            assert (archive / name).read_bytes() == data
    assert user_asset.read_bytes() == b"historical user data"
    assert (kernel.WORK / "SCOPE" / "runs" / "user" / "best_val.pt").is_file()


def test_success_publishes_small_dirs_hashes_and_preserves_user_assets(kernel, offline):
    kernel.main(shard=9, frame_limit=3, qp_grid=[27])
    names = [name for name, _ in offline.calls]
    assert names[-3:] == ["run_e1", "eval_e1", "report_e1"]
    assert names.index("build_deps") < names.index("build_tool")
    commands = dict(offline.calls)
    assert commands["build_deps"][0] == "apt-get"
    assert commands["build_tool"][0] == "gcc"
    assert all("bash" not in cmd and "echo" not in cmd for _, cmd in offline.calls)
    assert kernel.summary["status"] == "success"
    assert kernel.summary["tool_sha256"] == hashlib.sha256(b"compiled tool").hexdigest()
    assert kernel.summary["tool_source_sha256"] == hashlib.sha256(b"C tool source").hexdigest()
    for dest in (kernel.WORK / "out", kernel.WORK / "0000deliv"):
        assert {p.name for p in dest.iterdir()} == {"e1.csv", "quality.csv", "ceiling.csv",
                                                  "report.md", "summary.json"}
        summary = json.loads((dest / "summary.json").read_text())
        assert summary["status"] == "success"
        assert summary["result_rows"] == {"e1.csv": 2, "quality.csv": 2, "ceiling.csv": 2}
        assert (dest / "report.md").read_text() == "VERDICT: INCOMPLETE\n"
    assert (kernel.REPO / "user_source.py").is_file()
    assert (kernel.WORK / "SCOPE" / "runs" / "user" / "best_val.pt").is_file()
    assert not offline.scope_dir.exists()  # cleaned only after the mocked inference/report
    assert not (kernel.WORK / "real_e1").exists()
    assert (kernel.WORK / "logs" / "run_e1.log").is_file()
