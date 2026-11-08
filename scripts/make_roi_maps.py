#!/usr/bin/env python
"""Convert cached masks.npz to per-frame MB QP-offset maps for
tools/x264_roi_encode.c (A1 backend). Same masks as every arm (council R3-R1).

Usage: python scripts/make_roi_maps.py <masks.npz> <width> <height> <delta> <out_prefix>
Writes: <out_prefix>.txt (all frames, concatenated MB lines — the C tool
reads them sequentially per frame with fscanf).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope_vcm.masks import load_masks  # noqa: E402

MB = 16


def main() -> int:
    masks_npz, w, h, delta, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    masks = load_masks(Path(masks_npz))
    mbw, mbh = -(-w // MB), -(-h // MB)
    lines = []
    for t in sorted(masks):
        m = masks[t]
        vals = []
        for r in range(mbh):
            for c in range(mbw):
                blk = m[r * MB:(r + 1) * MB, c * MB:(c + 1) * MB]
                # object MB keeps base QP (offset 0); background pays +delta
                vals.append("0" if blk.size and blk.mean() >= 0.5 else str(delta))
        lines.append(" ".join(vals))
    Path(outp).write_text(chr(10).join(lines) + chr(10))
    print(f"{len(lines)} frames x {mbw * mbh} MBs -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
