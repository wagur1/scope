"""Real x264 tool regressions; skip when local development files are absent.

SCOPE_X264_ROI_TOOL may point to an already compiled real encoder. Otherwise
build in pytest's temporary directory using local cc/gcc and x264 only. No
mock encoder, downloads, installs, or experiment-results writes are involved.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "x264_roi_encode.c"


@pytest.fixture(scope="session")
def roi_tool(tmp_path_factory):
    supplied = os.environ.get("SCOPE_X264_ROI_TOOL")
    if supplied:
        tool = Path(supplied).resolve()
        if not tool.is_file():
            pytest.fail(f"SCOPE_X264_ROI_TOOL is not a file: {tool}")
        return tool
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        pytest.skip("native x264 smoke blocked: no local C compiler")
    build = tmp_path_factory.mktemp("x264-native-build")
    flags = ["-lx264", "-lm"]
    pkg_config = shutil.which("pkg-config")
    if pkg_config:
        pkg = subprocess.run([pkg_config, "--cflags", "--libs", "x264"],
                             capture_output=True, text=True, timeout=30)
        if pkg.returncode == 0:
            import shlex
            flags = shlex.split(pkg.stdout) + ["-lm"]
    suffix = ".exe" if os.name == "nt" else ""
    probe = build / "probe.c"
    probe.write_text("#include <stdint.h>\n#include <x264.h>\n"
                     "int main(void) { x264_param_t p; x264_param_default(&p); "
                     "x264_t *e = x264_encoder_open(&p); "
                     "if (e) x264_encoder_close(e); return 0; }\n")
    try:
        check = subprocess.run([compiler, str(probe), "-o", str(build / f"probe{suffix}"),
                                *flags], cwd=build, capture_output=True, text=True, timeout=60)
    except OSError as exc:
        pytest.skip(f"native x264 smoke blocked: compiler unavailable: {exc}")
    if check.returncode:
        pytest.skip(f"native x264 smoke blocked: local header/library probe failed: "
                    f"{check.stderr.strip()}")
    tool = build / f"x264_roi_encode{suffix}"
    compiled = subprocess.run([compiler, "-std=gnu99", "-Wall", "-Wextra", "-Werror", "-O2",
                               str(SOURCE), "-o", str(tool), *flags], cwd=build,
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return tool


@pytest.fixture
def inputs(tmp_path):
    raw = tmp_path / "input.yuv"
    # 18x18 exercises ceil-to-16 MB counts (four blocks, not one).
    raw.write_bytes((bytes([90]) * (18 * 18) + bytes([128]) * (18 * 18 // 2)) * 2)
    roi = tmp_path / "map.txt"
    roi.write_bytes(b"0 1.5 -2 0\n0 2.25 -1 0\n")
    out = tmp_path / "output.h264"
    return [str(raw), "18", "18", "29.97", "28", str(roi), "0", str(out), "2"]


def run_tool(tool, args):
    return subprocess.run([str(tool), *args], capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("index,value", [
    (1, "0"), (1, "-2"), (1, "17"), (2, "0"), (2, "19"),
    (1, "18junk"), (2, "2147483648"),
    (3, "0"), (3, "-1"), (3, "nan"), (3, "inf"), (3, "1e999"),
    (3, "30junk"), (3, "1e-999"),
    (4, "bad"), (4, "28.5"), (4, "-1"), (4, "52"),
    (6, "bad"), (6, "2.5"), (6, "2147483648"),
    (8, "0"), (8, "1"), (8, "-2"), (8, "two"), (8, "2junk"),
    (8, "3"), (8, "2147483648"),
])
def test_invalid_args_fail(roi_tool, inputs, index, value):
    inputs[index] = value
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0, result.stderr
    assert "encoded " not in result.stderr
    assert not Path(inputs[7]).exists()


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_wrong_argument_count_fails(roi_tool, inputs, mode):
    args = inputs[:7] if mode == "missing" else inputs + ["surplus"]
    assert run_tool(roi_tool, args).returncode != 0


@pytest.mark.parametrize("data", [
    b"0 0 0 0\n",                        # Missing frame.
    b"0 0 0\n0 0 0 0\n",                # Short first row.
    b"0 0 0\n0 0 0 0 0\n",              # Same total tokens, misaligned rows.
    b"0 0 0 0 0\n0 0 0\n",              # Reverse misalignment.
    b"0 0 0 0\n0 0 0 0 0\n",            # Extra entry.
    b"0 0 0 0\n0 0 0 0\n0 0 0 0\n",    # Extra frame.
    b"0 0 0 0\n0 0 0 0\n\n",           # Extra blank line is still a line.
    b"\n0 0 0 0\n",                      # Empty frame line.
    b"nan 0 0 0\n0 0 0 0\n",
    b"0 inf 0 0\n0 0 0 0\n",
    b"0 0 -inf 0\n0 0 0 0\n",
    b"1e39 0 0 0\n0 0 0 0\n",           # Finite double, overflowing float.
    b"0 0 1junk 0\n0 0 0 0\n",
    b"0 0 0 0\n0 0 0 0\ntrailing",
    b"0\x00junk 0 0 0\n0 0 0 0\n",
])
def test_malformed_map_fails_before_output(roi_tool, inputs, data):
    Path(inputs[5]).write_bytes(data)
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0, result.stderr
    assert "encoded " not in result.stderr
    assert not Path(inputs[7]).exists()


@pytest.mark.parametrize("size", [0, 486, 971, 973, 1458])
@pytest.mark.parametrize("explicit_n", [True, False])
def test_raw_size_must_be_exact(roi_tool, inputs, size, explicit_n):
    # One frame, incomplete frame, trailing byte, or explicit count mismatch.
    Path(inputs[0]).write_bytes(bytes(size))
    args = inputs if explicit_n else inputs[:-1]
    if size == 1458 and not explicit_n:
        # Three frames inferred correctly require three map lines.
        Path(inputs[5]).write_bytes(b"0 0 0 0\n" * 3)
        assert run_tool(roi_tool, args).returncode == 0
    else:
        result = run_tool(roi_tool, args)
        assert result.returncode != 0, result.stderr
        assert "encoded " not in result.stderr
        assert not Path(inputs[7]).exists()


@pytest.mark.parametrize("index", [0, 5])
def test_missing_input_fails(roi_tool, inputs, index):
    Path(inputs[index]).unlink()
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0
    assert not Path(inputs[7]).exists()


def test_output_open_failure_is_not_success(roi_tool, inputs, tmp_path):
    inputs[7] = str(tmp_path / "absent-parent" / "output.h264")
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0
    assert "encoded " not in result.stderr


def test_output_write_failure_is_not_success(roi_tool, inputs):
    if os.name == "nt" or not Path("/dev/full").exists():
        pytest.skip("local /dev/full write-failure device unavailable")
    inputs[7] = "/dev/full"
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0
    assert "encoded " not in result.stderr


@pytest.mark.parametrize("index", [0, 5])
def test_output_cannot_overwrite_input(roi_tool, inputs, index):
    original = Path(inputs[index]).read_bytes()
    inputs[7] = inputs[index]
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0
    assert Path(inputs[index]).read_bytes() == original


@pytest.mark.parametrize("index", [0, 5])
@pytest.mark.parametrize("alias_kind", ["dot", "parent", "hardlink", "symlink"])
def test_output_alias_preserves_both_inputs(roi_tool, inputs, tmp_path, index, alias_kind):
    source = Path(inputs[index])
    original = {i: Path(inputs[i]).read_bytes() for i in (0, 5)}
    if alias_kind == "dot":
        # Do not pass through Path: it normalizes away the literal /./.
        alias = str(source.parent) + os.sep + "." + os.sep + source.name
    elif alias_kind == "parent":
        (tmp_path / "subdir").mkdir()
        alias = str(tmp_path / "subdir" / ".." / source.name)
    else:
        alias = str(tmp_path / "alias-output.h264")
        try:
            if alias_kind == "hardlink":
                os.link(source, alias)
            else:
                os.symlink(source, alias)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"local {alias_kind} creation unavailable: {exc}")
    assert alias != inputs[index]
    assert os.path.samefile(source, alias)
    inputs[7] = alias
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0, result.stderr
    assert "encoded " not in result.stderr
    for i, content in original.items():
        assert Path(inputs[i]).read_bytes() == content
    assert Path(alias).read_bytes() == original[index]


def test_raw_alias_is_rejected_with_map_disabled(roi_tool, inputs):
    raw = Path(inputs[0])
    original = raw.read_bytes()
    inputs[5] = "-"
    inputs[7] = str(raw.parent) + os.sep + "." + os.sep + raw.name
    result = run_tool(roi_tool, inputs)
    assert result.returncode != 0, result.stderr
    assert raw.read_bytes() == original


def test_existing_unrelated_output_is_truncated(roi_tool, inputs):
    out = Path(inputs[7])
    result = run_tool(roi_tool, inputs)
    assert result.returncode == 0, result.stderr
    expected = out.read_bytes()
    out.write_bytes(b"obsolete output" * 10000)
    result = run_tool(roi_tool, inputs)
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == expected


@pytest.mark.parametrize("map_mode", ["float-crlf", "no-final-newline", "disabled"])
@pytest.mark.parametrize("explicit_n", [True, False])
def test_valid_input_encodes(roi_tool, inputs, map_mode, explicit_n):
    if map_mode == "disabled":
        inputs[5] = "-"
    elif map_mode == "no-final-newline":
        Path(inputs[5]).write_bytes(b"0 1.5 -2 0\n0 2.25 -1 0")
    else:
        Path(inputs[5]).write_bytes(b"0 1.5 -2 0\r\n0 2.25 -1 0\r\n")
    result = run_tool(roi_tool, inputs if explicit_n else inputs[:-1])
    assert result.returncode == 0, result.stderr
    assert "encoded 2 frames" in result.stderr
    assert Path(inputs[7]).stat().st_size > 0


def test_roi_offsets_are_not_silently_disabled(roi_tool, tmp_path):
    import random

    raw = tmp_path / "noise.yuv"
    rng = random.Random(7)
    raw.write_bytes(b"".join(bytes(rng.randrange(16, 236) for _ in range(64 * 64)) +
                            bytes([128]) * (64 * 64 // 2) for _ in range(8)))
    roi = tmp_path / "offsets.txt"
    streams = {}
    for mode, offset in (("disabled", None), ("zero", "0"), ("roi", "12")):
        out = tmp_path / f"{mode}.h264"
        if offset is not None:
            roi.write_text((" ".join([offset] * 16) + "\n") * 8)
        result = run_tool(roi_tool, [str(raw), "64", "64", "30", "28",
                                    "-" if offset is None else str(roi), "0", str(out), "8"])
        assert result.returncode == 0, result.stderr
        streams[mode] = out.read_bytes()
    assert streams["zero"] == streams["disabled"]
    assert len(streams["roi"]) < len(streams["zero"])


def test_rational_fps_and_frame_count_in_bitstream(roi_tool, inputs):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("local ffprobe unavailable for bitstream metadata check")
    result = run_tool(roi_tool, inputs)
    assert result.returncode == 0, result.stderr
    probe = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
                            "-of", "json", inputs[7]], capture_output=True, text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (18, 18)
    assert int(stream["nb_read_frames"]) == 2
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    assert numerator / denominator == pytest.approx(29.97, abs=1e-6)
