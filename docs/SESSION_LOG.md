# Session Log — 2026-11-07 (Kaggle execution day)

## What ran (kaggle pool: 21 accounts)

| Kernel | Status | Result |
|---|---|---|
| scope-e1-smoke v4 | 21/21 COMPLETE | **run_e1 + eval_e1 PASS on all 21** (synthetic grid, x265); report correctly refuses: quality axis dead (YOLO sees nothing on noise+red-box content). **Determinism: 0/21 accounts differ in bitrate >0.5 kbps (same seed+encoder)** — cross-machine determinism verified at scale |
| scope-roi-hunt v6 | 3/3 COMPLETE | Decisive findings below |
| scope-real-e1 v4 | 21 RUNNING (launched 86238fd) | run_e1 **PASS on real MOT17** in v3 (63/63 rows); v4 fixes eval (detector_classes wiring). Results land ~30 min after launch: `python scripts/kaggle_pool.py status kaggle_pool.json --slug scope-real-e1` then `fetch` |

## ROI hunt (Bottleneck 1) — what is now KNOWN (not guessed)

1. **`quant_offsets` EXISTS in x264.h (3 hits); `quantOffsets` EXISTS in x265.h (1 hit); `qp-delta-map` CLI option does NOT exist (0 hits)** — the C-API is the only stock-toolchain path. (out/roi_hunt.json, committed artifacts)
2. x264 C-API sign-test: compiles and runs, but **offsets are silently ignored under CQP AND under CRF+aq-mode=1** (bitrate identical to base, ratio=1.0 twice). Hypotheses for next iteration: (a) offsets interact with mb-tree/lookahead — try `b_mb_tree=0`; (b) offsets array must be freed/reprovided per frame; (c) x264 requires `param.analyse.b_weighted_pred`-style gates — read x264's ratecontrol.c `x264_adaptive_quant_frame` directly for the exact gate condition.
3. x265 C-API: compiles; container crashes solved one at a time (fps via param_parse ✓, numa-pools=none ✓); remaining crash: `double free or corruption` — **x265 takes ownership of pic.quantOffsets and frees it** (documented behavior) → must allocate a fresh array per frame (or not reuse after first encode).
4. Fallback if both C-API paths stall: build VVenC on Kaggle and check `--roi`/`--qp` map support empirically (cmake build ~5 min), or patch VTM per MPEG VCM EE practice (m64712).

## SCOPE integration (Bottleneck 2) — READY

Adapter verified against upstream `scope/inference.py`: `load_scope_model` + `apply_scope_bgr(img_bgr, ckpt, codec_family, severity)` (BGR uint8 in/out, families diff_jpeg/bmshj/bpg_like, severity ∈ [0,1]). Weights: Google Drive folder `1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES` → `runs/<exp>/best_val.pt`, fetched in-kernel via `python -m gdown --folder`. Driver wires A3 automatically when the checkpoint lands (graceful skip otherwise). Detectron2 may be required by upstream imports — if so, add the pip line to the driver (build ~5 min on Kaggle).

## Real data (Bottleneck 3) — DONE for E1 scope

`nhnhunhch/mot17-coco` (public dataset) = MOT17 + COCO annotations. `kaggle/real_e1_kernel.py`: 7 FRCNN sequences × 150 frames @ 960×540, x264 veryfast CQP {27,32,37}, arms A0/A2/A2c, YOLOv8n person-class scorer vs pedestrian GT (scaled boxes). v3 proved run_e1 end-to-end on real data; v4 (commit 86238fd) fixes the eval step (detector_classes wiring — YOLO emitted vehicle classes, the single-class GT assert fired correctly by design).

## Bug classes caught by running on Kaggle (each invisible locally)

1. Mount-path assumptions (dataset root varies) → runtime discovery
2. sys.path for in-kernel imports
3. Config→code wiring gaps (masks_dir, detector_classes)
4. Kernel-output API truncation on large file trees → out/ packaging + heavy-dir cleanup
5. pycocotools requires annotation['id']
6. GT images/annotations id consistency (shift for uniqueness broke alignment)

## Next session queue (ordered)

1. Fetch real-E1 v4 (should be COMPLETE): pooled BD(A0→A2) on REAL MOT17 content + 21× determinism — first real numbers of the project
2. Hunt iteration 7: x264 b_mb_tree=0 test; x265 per-frame offsets allocation → A1 sign test green
3. Implement the winning path in codec.py (ctypes wrapper for x264/x265 C-API), re-run check_encoders self-test, enable A1/A4
4. Smoke v5 with SCOPE A3 enabled (weights via gdown; detectron2 if needed)
5. Scale real-E1: full QP grid + full-length sequences, one sequence per account × 3 replicates (pool parallelism)
