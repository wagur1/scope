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

## FIRST REAL-NUMBER RESULT (2026-11-07 late) — real MOT17, x264 CQP, YOLOv8n-person

Per-seq (ceiling-normalized): BD(A0→A2 selective blur) = −5.1% … −82.9%, median −49.5% (n=7); blur-everything control even larger (−70…−77%, n=2 valid). **BUT η = 0.003…0.298 — F2 FIRES DECISIVELY** (gate ≤ 0.01): uncompressed pre-edited mAP drops 3–30 points → the raw BD-rate gains are dominated by content destruction, not compression. The blur-everything control confirms the mechanism: biggest "savings" + worst ceilings.

Reading: exactly the failure mode the ceiling-normalization was built to expose (council R3-fatal-2). The unbounded blur fallback is dead as a *compression* claim; it is a valid *accuracy-recovery* artifact only. Implications: (a) bounded edits (SCOPE α=0.09/β=0.03, arm A3) are the correct next test — bounds exist precisely to cap η; (b) the decisive thesis comparison remains pre-edit vs **A1 ROI delta-QP** (blocked on the C-API sign test, see hunt log); (c) determinism verified again: 0/63 rows differ across 3 accounts on real encodes.

GPU real-E1 (`scope-real-e1-gpu`, T4-or-P100 recorded per-kernel in summary["gpu"]) launched on all 21 accounts — GPU-vs-CPU scorer deltas expected (ceilings recomputed per run, so η/BD stay internally valid).

## Scorer alignment (reviewer note: VCM papers use YOLOv5 + Faster R-CNN)

detect.py now dispatches backends: yolov5su.pt (ultralytics — Dual-Region ITC-CSCC'25, Shindo CCNC'24 standard) and frcnn_resnet50_fpn_v2 (torchvision pretrained — Faster R-CNN family, the MPEG VCM CTC anchor family; CTC-exact Detectron2 X101 is the full-res post-gate upgrade). Masks also switch to yolov5su. Both real-E1 variants relaunched on all 21 accounts (CPU + GPU). The completed yolov8n run stays valid as a scorer-robustness data point: the F2 content-destruction finding should hold under any competent scorer if it is real — next fetch verifies.

## Shard fleet launched (user: "chia 21 shard ra mà làm", "nếu chiếm thì bỏ qua")

21 shards = 7 sequences × 3 replicates, each shard: 1 sequence × full 9-QP grid {22..42} × 300 frames × arms A0/A2/A2c, YOLOv5su masks + FRCNN scorer, on GPU. Pushed as per-account wrappers (kaggle_pool --shard generates wrapper_i with main(shard=i, frame_limit=300, qp_grid=full)). 19 shards running on GPU; line344 + nguoingoaihanhlang SKIPPED (their GPU quota occupied by older kernels — user: skip, no CPU fallback). Replicate triplets (shard i, i+7, i+14) give the cross-account determinism check on the full grid.

Bugs fixed to get there: argparse % escaping, duplicated --shard entry from double-patch, wrapper generation. Fleet also still running: 21 CPU FRCNN full-runs (150 frames, 3 QP) + 21 GPU yolov8-scorer runs (round-4 code).

Morning queue unchanged: fetch shards (`--slug scope-real-e1-shard`), aggregate per-sequence full-grid BD + replicate determinism; hunt iteration 7 (x264 b_mb_tree=0, x265 per-frame offset allocation); SCOPE A3 smoke.

## QP grid decision (owner): 5 CTC levels primary

Owner (2026-11-07): consider 5 QP levels. Primary analysis = the pre-registered VCM CTC set {22, 27, 32, 37, 42}; the 4 intermediates {24, 29, 34, 39} collected by the in-flight 9-point shards are demoted to sensitivity-only. Running shards are a superset — no re-push, no wasted compute; report_e1 tables slice the 5 CTC rows. Wrapper template updated for future pushes.

## SHARD RESULTS (19/19 COMPLETE — T4 confirmed, F2 scorer-robust)

All 19 shards: run_e1 + eval_e1 PASS (~50 min each). **GPU probe: Tesla T4 on every shard** (user's T4 question answered empirically — metadata API has no type field, probe is the proof).

**Replicate determinism: 0 bitrate mismatches across all 7 sequence triplets** (19 shards, 3 accounts per sequence, full 9-QP grid × 300 frames).

**F2 fires 7/7 under the Faster R-CNN scorer** (ceiling-normalized, 5 CTC levels): η = 0.028 (MOT17-09) → 0.516 (MOT17-13), median raw BD −62.1%. The yolov8n run fired 6/7 with η ≤ 0.298; FRCNN fires harder. Conclusion is now **scorer-robust**: unbounded blur pre-editing's raw BD-rate "gains" are content destruction on both scorers, on real MOT17, across the full CTC grid, replicated 3× per sequence with zero drift. MOT17-13 is the extreme case: half the task accuracy destroyed pre-encode.

Status of the pre-registered question: unbounded pre-editing is dead as a compression claim (F2). The open tests remain A1 (ROI delta-QP baseline — hunt iteration 7 pending) and A3 (bounded SCOPE edits — adapter ready). The framework (ceiling-normalization + gates) worked exactly as pre-registered on every run.

## GPU full-runs aggregated (21 accounts, yolov8n-on-GPU) — CPU runs dropped per owner

Determinism: 0 mismatches × 21 accounts. Per-seq BD(A0→A2, 5 CTC... QP{27,32,37}): −38.5%…−87.0%, median −67.3%; **F2 fires 7/7** (η 0.036–0.493). Cross-scorer/device F2 tally: yolov8-CPU 6/7, yolov8-GPU 7/7, FRCNN-GPU(shards) 7/7 → content-destruction finding is scorer- AND device-robust.

**Policy (owner): GPU-only for all future eval.** CPU FRCNN full-runs OOM'd mid-eval (no eval log → hard kill) and are abandoned; CPU role going forward = none for kernels (local CPU only for analysis).

## ROI hunt rounds 7-12 (Gemini 4 Argon consult + patched-build tracks) — state

Consulted Gemini 4 Argon (Medium) via Antigravity CLI (`agy --print --model gemini-4-argon-medium`) — delivered source-level analysis: x264 offsets flow through `x264_adaptive_quant_frame` (needs aq-mode≠0, and slice-QP lock via `pic.i_qpplus1=qp+1`); x265 needs CRF+aqMode=1+aqStrength=0+cuTree=0+qgSize=16+`pic.forceqp`, offsets sized by 16×16 blocks (our 64×64 sizing was a bug), ffmpeg ROI side-data hits the SAME gates.

Round 9 (Argon rx applied): x264 base bitstream CHANGED (config bites) but ±10 offsets still byte-identical → `prop.quant_offsets` inert on the Ubuntu apt libx264 build. Round 10-11 (patched x265): first patch anchor stale (gate line moved to param.cpp/encoder.cpp `p->rc.aqMode`), fixed with dynamic grep; build produced lib but **harness itself corrupts the heap at exit in BOTH base and offsets runs** (double free, offsets-independent — bug in our C test harness lifetime, not the library gate).

Round 12 (final layout, clean rewrite): stock x264/x265 + patched x264 + patched x265 + **kvazaar stock --roi** tracks. Result invalidated by environment: the same apt transaction that adds kvazaar removed/broke libx264-dev (x264.h vanished; kvazaar binary absent on this Ubuntu release). Stock-x264 track therefore untested this round — and x264 MASTER no longer contains the `i_aq_mode = X264_AQ_NONE` gate Argon cited (removed upstream), so **x264-master C-API may work where the old apt build does not** — untested.

Next-session hunt queue (round 13):
1. Separate apt transactions; verify `test -f /usr/include/x264.h` between steps
2. Stock x264 sign test on apt build + x264-master source build (gate gone upstream → possibly fixed)
3. Kvazaar: install alone (check package exists: `apt-cache policy kvazaar`), else build from ulfosson/kvazaar master
4. x265 harness crash: isolate with a minimal main (no offsets, no custom planes — copy x265's own example) to find the harness bug before blaming the library
5. On any green sign test → implement A1 in codec.py, relaunch real-E1 shards with A1/A4 arms

## External dual review (2026-11-07): Gemini 4 Argon Medium (agy CLI) + GPT-6 Astra High ($xhigh, workflow)

Top findings affecting the thesis:

1. **[BOTH] η/ceiling-normalization estimand correction**: dividing by arm-specific ceilings rewards lower attainable accuracy; the claim that editing makes absolute-mAP BD "formally invalid" is too strong (fixed detector + same GT labels). → Keep normalized endpoint as arm-relative retention; ADD direct A-vs-A1 BD on a shared absolute-mAP interval + rates at fixed absolute-mAP targets as the confirmatory contrast (Astra #1, #6; Argon concurs via estimand note).
2. **[BOTH] A1 sign-test harness bugs invalidate the "DEAD" verdict**: x264 CRF+aq still ratio 1.0 BUT Argon identified i_qpplus1 (round 9 addition) overrides QP AFTER the AQ path — remove it; Astra adds: probes stop after one flush (delayed frames not drained), x264 codec.py branch silently ignores roi_map (A1 could pass vs a nonexistent intervention), patched x264 probes must compile against the source tree's x264.h not apt's. x265: picture_free on stack picture + 16×16-vs-CTU size mismatch. → A1 "DEAD" verdict WITHDRAWN pending corrected harness.
3. **[Astra HIGH] report_e1.py blocks**: thesis-arm selection silently drops A2 when A3 present (co-primary rule violated); A4 ceiling KeyError; min() can skip NaN (violates NaN hard-fail); 9-QP not filtered to 5 CTC for headline; merge quality_x/quality_y.
4. **[BOTH] MOT17 has only 7 sequences — and Argon notes G-1's n≥8 rationale is mathematically wrong anyway** (5 untied differences all favoring → exact tail 1/32 < 0.05). Need ≥3 more independent sequences (MOT20/COCO-video); machine replication counts only as technical replication.
5. **[Astra HIGH] A3 matched-mask confound**: SCOPE uses its own attention while A1/A2 use cached YOLO masks → A3-vs-A1 would confound mask with edit. Fix: export A3's attention mask for its paired ROI baseline or fixed-mask A3 ablation.
6. **[Astra HIGH] α/β bounds do NOT cap total content change**: (1−A)·(Blur(I)−I) is uncontrolled → A3 can still fire F2; evaluate A3 ceilings, don't assume. F2 must be presented as failure of the task-utility preservation requirement, NOT a causal decomposition of savings.
7. **[Astra] Preregistration honesty**: 9→5 QP demotion happened after pilot results were visible; PROTOCOL still says 9 → publish an amendment ledger; reconcile documents; use a locked untouched cohort for the confirmatory A1/A3 decision.
8. **[Astra] κ definition flawed** (|·| counts regressions as savings; multiplicative savings combine sub-additively) → signed factorial interaction on common-support log rates before E2.
9. **[Argon] Missing cheap baselines**: dilated/feathered mask blur (context-loss vs background-destruction), spatial downscaling (simplest VCM baseline).
10. **[Astra] scorer bug**: FRCNN internal box_score_thresh=0.05 discards detections below it before our conf=0.001 filter → claimed low-threshold protocol not actually in effect; also COCO class-id +1 mapping wrong for gapped ids; decoded-frame existence never validated; useCats=0 alone insufficient for class-agnostic eval.

Paper framing (Argon): "The Unbounded Pre-Editing Mirage in VCM" — 2-panel figure (raw RD illusion vs ceiling-normalized truth + η disclosure). Astra cautions: present as failure of task-utility preservation, not causal decomposition.

Priority queue (merged): (1) fix C harness per #2 + rerun sign test; (2) fix report_e1 blocks #3 + add ≥3 sequences; (3) wire dilated-mask + downscale baselines; (4) A3 η check + matched-mask fix; (5) amendment ledger for QP demotion.

## SCOPE A3 BOUNDED RESULT (2026-11-07, scope-real-e1-scope run, MOT17-02, x264, YOLOv8n)

| Arm | ceiling mAP | η | BD(A0→arm) |
|---|---|---|---|
| A0 anchor | 0.2272 | — | — |
| A2 unbounded blur | 0.1927 | 0.152 ❌ | −78.6% |
| A2c blur-all | 0.0648 | 0.715 ❌ | −90.1% |
| **A3 SCOPE bounded (α=0.09/β=0.03, diff_jpeg, sev 0.5)** | **0.2207** | **0.028** | **−2.52%** |

**Đọc kết quả (single-sequence pilot):** bounds SCOPE giảm content destruction 5.4× (η 0.152→0.028; giữ 96.9% ceiling vs 84.9%) — đúng như thiết kế α/β — nhưng vẫn vượt gate 0.01 (mất 2.8 điểm mAP), và honest BD chỉ −2.5% so với fake −78.6% của unbounded. Toàn bộ dải kết quả chạy trên cùng sequence, cùng encoder, cùng scorer: fake savings tỷ lệ thuận với destruction; bounded editing → gain thật nhỏ. Còn thiếu: n≥8 sequences, A1 baseline, severity-mapping theo QP (scope_severity hiện cố định 0.5 — Astra finding), matched-mask cho A3-vs-A1.

Cron fleet paused (tất cả slugs terminal).

## EXTERNAL REVIEW received (2026-11-07) — 4.5/10 pre-submission assessment, commit 032d3fa

Independent reviewer (docs/REVIEW-EXTERNAL.md) REPRODUCED 6 P1 bugs with synthetic fixtures (no video needed): report merge KeyError 'quality'; co-primary A2/A3 not both evaluated (A2-pass fixture still NO-GO); missing-A1 → NO-GO misread; x264 wrapper silently ignores ROI (mock: identical command); A4 missing ceiling KeyError; 9-vs-5 QP inconsistency. Plus P2: join one-to-one not enforced, BD dedupe ValueError on equal-rate points, low-delay GOP not actually pinned, resume cache not identity-checked, Wilcoxon n≥8 rationale wrong (5 diffs → p=0.03125, matches Argon).

Reviewer verdict: 4.5/10 as submission evidence; "framework design good (8/10 design), evidence packaging weak". All P1s are real and unfixed at current HEAD → fixing now, with regression tests mirroring their fixtures.

## P1 FIXES from external review (same session, 20/20 tests)

- **P1-01+P2-01** (report_e1.load): placeholder quality column dropped before merge; merge validate="one_to_one"; duplicate/missing keys rejected explicitly (reviewer fixture reproduced the old KeyError + the 120→120 row-count hole)
- **P1-02+P1-03** (report verdict): co-primary arms evaluated independently with per-arm G-eta/G-1; verdict taxonomy INCOMPLETE / INCONCLUSIVE / NO-SUPPORT / GO — missing A1 or n<8 can no longer print NO-GO/publish-F1
- **P1-04** (codec.py): x264 ROI request now raises NotImplementedError (was silent no-intervention encode)
- **P1-05** (report ceiling_for): A4 ceiling inherits A2 (identical edited source), printed note
- **P1-06** (load primary_qps): headline restricted to config primary_qps; excluded rows labelled sensitivity-only
- **P2-02** (bd.py): split _clean_for_rate / _clean_for_quality — equal-rate plateaus no longer averaged away; equal-quality duplicates no longer crash bd_quality
- metrics.py: n≥8 rationale corrected (registered power rule, NOT math necessity — 5 untied diffs → p=1/32)
- Regression tests mirror the reviewer's fixtures: 20/20 pass

## Argon verification gates (3 rounds) — GO-KAGGLE

Round 1: C harness (x264/x265 round-14) FIXED đúng; report/eval còn 5 bug (verdict taxonomy conflates INCONCLUSIVE/NO-SUPPORT, arm_gaps zip misalignment, metrics pre-filter mutation + fit dedupe, __all__ stale, ultralytics frames check, dilate_px not wired). Round 2: 5 fixes applied. Round 3: bd __all__, detect exact img-id set check, x265 zero-latency harness (bframes=0/lookaheadDepth=0/bRepeatHeaders=1). Round 4: **GO-KAGGLE**. All fixes committed through 71f5ef5; 20/20 tests.

## Hunt round 14 (post GO-KAGGLE, verified harness) — DECISIVE negative

With the review-corrected harness (no i_qpplus1, full drain, zero-latency x265): x264 stock, x265 stock AND x265-patched (AQ gate patched, CRF+aq1+strength0+cuTree0+qgSize16+forceqp) ALL produce byte-identical bitstreams with/without ±10 quantOffsets. x265 stderr even shows "consecutive B-frames: 100.0%" despite p->bframes=0 — param application itself is suspect on this path, or the quantOffsets feature is dead in current master. Empirical conclusion after 8 rounds, 3 independent harness generations: **the quantOffsets C-API route does not work on these toolchains** (x264 apt 0.164, x265 master-clone + apt).

Remaining A1 routes (owner decision):
- R15a: kvazaar from source (ulfosson/kvazaar, native --roi, ~10 min build) — cheapest
- R15b: x265 master grep for qp-delta-map / newer API (untested hypothesis)
- R15c: VTM source patch per MPEG VCM EE practice (heavy, CTC-exact)
- R15d: pre-registered amendment — substitute the A1 baseline with "per-frame zones" (temporal) documented as a weaker bound, or drop A1 and reframe E1 per the external-review option (A2-only study without the ROI claim)
