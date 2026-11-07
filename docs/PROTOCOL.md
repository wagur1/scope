# Experiment Protocol — E1 (Decisive) and E2 (Gated)

Pre-registered before any results are seen. Thresholds come from `MATH.md` §9; changing them post-hoc is prohibited.

## E1 — Does pixel pre-editing beat lossless ROI delta-QP?

**Cost:** zero training. 1 GPU for pre-edit/masks, CPU for encoding. ~6–10 weeks.

### Data
- **Sequences (video):** MOT17 train split subset + COCO-derived short clips (config-driven; `configs/e1_decisive.yaml`). CTC-adjacent targets: MOT16/17/20 for tracking (JDE anchor), COCO val2017 sequence-ized for detection.
- **Masks:** predicted once per sequence by a public detector (YOLO-family via `ultralytics`, or any box source); cached to `masks/{seq}.npz`. **Same mask file feeds every arm.** GT boxes are permitted ONLY for a diagnostic upper-bound arm, never in headline tables.

### Arms
A0 anchor · A1 ROI delta-QP (δ ∈ {4,8,12,16}, strongest-δ rule for the gate) · A2 training-free pre-edit (**co-primary thesis arm, α=β=0**) · A3 SCOPE per-frame (**co-primary thesis arm, learned α=0.09, β=0.03**; full F1 falsification requires A3 evaluated alongside A2) · A4 pre-edit + ROI (diagnostic) · **A2c blur-everything control** (mask ≡ 0, mechanism isolation). QP grid {22, 24, 27, 29, 32, 34, 37, 39, 42} (CTC + intermediate mid-QPs — the gain window may be narrow), low-delay. Minimum **8 sequences** (hard gate), ≥10 preferred; per-domain (MOT vs COCO) medians reported via `report_e1.py --config`.

### Encoders
| Encoder | Use | Status |
|---|---|---|
| x265 (fixed-QP `qp=`, `aq-mode=0`, `cutree=0`; per-LCU map `qp-delta-map` **NOT in public x265** — verified against x265cli.h/ffmpeg libx265.c 2026-11) | main E1 codec; A1 needs a map-capable build | `scripts/check_encoders.py` self-test is the gate (bitrate-ordering pins sign empirically) |
| x264 (`-qp`, CQP) | reproduction of Zhao DCC deploy settings | stable |
| VTM (VVC CTC low-delay) | CTC anchor for E2 | build from VTM; per-CTU ROI path **VERIFY** (slice/QP-offset mechanism differs by version) |
| VTM+NNVC-4.0 | E2 additivity arm | build from JVET NNVC software |

**Encoder policy (pre-registered):** every x265 arm runs identical params (CQP, AQ off, cutree off) so the ROI map applies nominal offsets; lossless ceiling encodes use `lossless=1, yuv444p` **identically for m\*_E and m\*_anchor** (MATH.md §4); encoder version strings are written into the run manifest (determinism is per-version — presets/threads can perturb output).

### Procedure (per sequence × arm × QP)
1. Pre-edit frames (A2/A3/A4) — deterministic, fixed seed where sampling exists; σ = 8.0 px fixed by MATH.md §2 (never tuned post-hoc).
2. Encode with the arm's codec config; record **bitrate** and **decode time**; abort on any "unknown option" in encoder logs (silent degradation is worse than failure).
3. Decode, run frozen task model (checkpoint pinned in manifest), record mAP (detection) or IDF1 (tracking).
4. Record ceilings: accuracy of the **losslessly-encoded** edited sequence (A2/A3) and of the original (A0) — the identical lossless path for both (MATH.md §4).
5. One row per (seq, arm, QP) → `results/e1.csv`; ceilings → `results/ceiling.csv`; the runner aborts the report if any rate row lacks a quality row.

### Analysis (`scripts/report_e1.py`)
- BD-rate per sequence (PCHIP, common normalized-quality interval — `scope_vcm/bd.py`).
- Ceiling drop η per (seq, method); **η > 0.01 flags the arm** (F2).
- Thesis statistic: median per-sequence BD gap A2/A3 vs A1 + one-sided Wilcoxon (α=0.05, n = #sequences).
- Per-sequence distributions printed with worst case; no aggregate-only tables.
- **Go/No-Go verdict block** evaluating gates G-η and G-1 mechanically.

### Pre-registration snapshot
Before the first run, this file and the config are committed; the run timestamp and git SHA are written into `results/run_manifest.json`.

## E2 — Additivity with in-loop NN tools (gated on E1 pass)

1. Rebuild A3 under VTM low-delay (per-CTU ROI path verified first).
2. Arms: VTM · VTM+NNLF · VTM+A3 · VTM+A3+NNLF · (optional) VTM+FGS arms.
3. Compute κ per `MATH.md` §7 with bootstrap CIs (10k resamples over sequences).
4. Gates: G-2 (|κ| ≥ 1% median) decides the interaction paper; null published either way.

## E2t — Temporal pre-editing (gated on E2 pass; the video paper)

Only after both gates. GOP-matched differentiable VTM proxy (hierarchical-B reference structure reproduced; proxy-vs-VTM rate error must be < 10% on held-out clips — council R2 gate). Three training arms per `MATH.md` §8; bar: +3% BD-rate and ID-switch reduction vs per-frame. Compute warning: council R2 estimated ~2 weeks × 8 A100-class GPUs for proxy training at video scale — do not start without gate passes and budget.

## GAP-10 — Proxy-fidelity pivot study (activated if F1 fires on E1)

If both A2 and A3 fail Gate G-1 against A1 (in-codec ROI delta-QP), pixel pre-editing does not beat lossless in-codec bit allocation under standard inter coding. The program then executes the **GAP-10 pivot**:
1. **Paper 1 (from E1):** Publish the E1 null result + matched-mask ROI delta-QP baseline + symmetric ceiling-normalized BD accounting (`m*_E`, `η`, truncated absolute BD).
2. **Paper 2 (GAP-10 study):** Quantify why neural pre-editors underperform on real video codecs — measure proxy-vs-real-codec rate rank correlation (Spearman ρ), gradient cosine similarity, and BD-rate sign-flip rates across 3 differentiable proxy families: (a) factorized intra proxy (SCOPE-style), (b) 2-frame optical-flow inter proxy (Zhao DCC-style), and (c) GOP-matched hierarchical-B proxy (NHK PCS 2024-style) against real x265/VTM low-delay encodes.

## Threats to validity (from council)

1. **ROI delta-QP erases the premise** → F1, handled by design (it IS the thesis test).
2. **Narrow QP window** (mid-QPs only) → per-QP tables mandatory; aggregate BD reported alongside, never alone.
3. **Proxy–codec sign flip** → E2t uses GOP-matched proxy; E1 is proxy-free (real codecs) by design.
4. **Mask source leakage** → predicted masks only in headline; GT-mask arm labeled diagnostic.
5. **Single-seed noise** → encoders are deterministic for fixed settings; detection eval is deterministic; bootstrap CIs over sequences for all aggregates.
6. **FCM makes the niche obsolete** → every paper frames against FCM explicitly (council R1): "what does bounded pixel editing buy when feature coding exists".

## Directory contract

```
data/{seq}/frames/*.png       # extracted input frames
masks/{seq}.npz               # per-frame masks (shared by all arms)
edited/{method}/{seq}/*.png   # pre-edited frames per method
enc/{arm}/{seq}/{qp}/         # bitstreams + decode outputs
results/e1.csv                # one row per (seq, arm, qp)
results/run_manifest.json     # git SHA, timestamps, config hash
```
