#!/usr/bin/env python
"""Pack %04d.png frames into a raw I420 .yuv file for the C ROI encoder."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    frames_dir, outp = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(frames_dir.glob("*.png"))
    assert files, f"no pngs in {frames_dir}"
    im0 = Image.open(files[0]).convert("RGB")
    w, h = im0.size
    with open(outp, "wb") as f:
        for fp in files:
            im = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32)
            # BT.601 limited-range RGB -> YUV420, vectorized
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            y = 0.257 * r + 0.504 * g + 0.098 * b + 16
            u = -0.148 * r - 0.291 * g + 0.439 * b + 128
            v = 0.439 * r - 0.368 * g - 0.071 * b + 128
            y = y[::2, ::2] if False else y  # keep full-res luma
            f.write(np.clip(y, 0, 255).astype(np.uint8).tobytes())
            u2, v2 = u[::2, ::2], v[::2, ::2]
            f.write(np.clip(u2, 0, 255).astype(np.uint8).tobytes())
            f.write(np.clip(v2, 0, 255).astype(np.uint8).tobytes())
    print(f"{len(files)} frames {w}x{h} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
