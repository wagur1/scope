#!/usr/bin/env bash
# ROI delta-QP hunt — decides Bottleneck 1 empirically on the host toolchain.
# Output: out/roi_hunt.json (consumed by codec.py implementation choice).
set -u
cd /kaggle/working
mkdir -p out logs
R=out/roi_hunt.json
echo '{}' > "$R"

apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq libx264-dev libx265-dev build-essential >/dev/null 2>&1
echo "toolchain installed"

H264_H=$(find /usr/include -name x264.h | head -1)
H265_H=$(find /usr/include -name x265.h | head -1)
echo "headers: $H264_H $H265_H"

# --- header probes -----------------------------------------------------------
X264_QO=$(grep -c "quant_offsets" "$H264_H" 2>/dev/null || echo 0)
X265_QO=$(grep -c "quantOffsets" "$H265_H" 2>/dev/null || echo 0)
X265_QPDM=$(grep -c "qp-delta-map\|qpDeltaMap" "$H265_H" 2>/dev/null || echo 0)
python3 - "$R" <<EOF
import json
r = json.load(open("$R"))
r["x264_header_quant_offsets"] = int("$X264_QO")
r["x265_header_quantOffsets"] = int("$X265_QO")
r["x265_header_qp_delta_map"] = int("$X265_QPDM")
json.dump(r, open("$R", "w"), indent=2)
EOF

# --- compile + sign tests ----------------------------------------------------
W=160; H=120; N=24; QP=28
compile_run() { # $1=lib(x264|x265) $2=use_off $3=outpath
  gcc -O2 -o "hunt_$1" "kaggle/roi_hunt_$1.c" -l"$1" -lm 2>>logs/roi_hunt_gcc.log
  "./hunt_$1" $W $H $N $QP "$2" "$3"
}

if [ "$X264_QO" -gt 0 ]; then
  compile_run x264 0 out/x264_base.bin && compile_run x264 1 out/x264_off.bin
  B=$(stat -c%s out/x264_base.bin); O=$(stat -c%s out/x264_off.bin)
  python3 - "$R" "$B" "$O" <<'EOF'
import json, sys
r, b, o = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
r["x264_test"] = {"base_bytes": b, "offset_bytes": o, "ratio": round(o / b, 4),
                  "works_and_sign_positive": bool(o < 0.95 * b)}
json.dump(r, open(sys.argv[1], "w"), indent=2)
EOF
  echo "x264 test: base=$B off=$O"
else
  echo "x264: quant_offsets NOT in header"
fi

if [ "$X265_QO" -gt 0 ]; then
  compile_run x265 0 out/x265_base.bin && compile_run x265 1 out/x265_off.bin
  B=$(stat -c%s out/x265_base.bin); O=$(stat -c%s out/x265_off.bin)
  python3 - "$R" "$B" "$O" <<'EOF'
import json, sys
r, b, o = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3])
r["x265_test"] = {"base_bytes": b, "offset_bytes": o, "ratio": round(o / b, 4),
                  "works_and_sign_positive": bool(o < 0.95 * b)}
json.dump(r, open(sys.argv[1], "w"), indent=2)
EOF
  echo "x265 test: base=$B off=$O"
else
  echo "x265: quantOffsets NOT in header (qp-delta-map occurrences: $X265_QPDM)"
fi

# --- PyAV presence (fallback transport) --------------------------------------
python3 -c "import av; print('pyav', av.__version__)" > logs/pyav.txt 2>&1 || echo "pyav missing" > logs/pyav.txt

echo "=== RESULT ==="
cat "$R"
