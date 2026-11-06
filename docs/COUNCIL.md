# Council Verdict — 2026-11-07 (4 reviewers, adversarial format)

Direction reviewed: "Temporally consistent content-aware pre-editing for VCM".
Full context: ARA rigor review v2 (`rigor-gaps-neural-codec-preprocessing.md`, workspace) + full-text reads of 17 papers.

## Verdicts

| Reviewer | Verdict | Heaviest objection |
|---|---|---|
| R1 — MPEG/JVET standardization engineer | MODIFY | FCM (ISO/IEC 23888-4) reports ~85% bitrate reduction vs VCM anchor; the pixel domain's only moat is a standards-compliant bitstream with human-tolerable output and zero decoder change. Bounded distortion + human-quality floor + dual-axis reporting required. |
| R2 — ML methodologist | MODIFY | Temporal consistency may be emergent from an inter-frame proxy loss (Zhao DCC already trains through an inter proxy); must be proven by quantified ablation, bar +3% BD-rate, else no paper. |
| R3 — adversarial Reviewer-2 | REJECT as proposed | FATAL-1: VTM ROI delta-QP (MPEG EE m64712/PUT, VCMRS 0.6.1) achieves the same goal losslessly, zero content modification — unbeaten, the paper has no thesis. FATAL-2: BD-rate is ill-defined when pre-editing changes the signal itself → ceiling-normalized mAP required. |
| R4 — research strategist | Gate it | ByteDance is sweeping the space (arXiv 2512.15270/15331, adjacent IDs); scoop window 6–12 months; DCC 2027 deadline passed. Video proxy training ≈ 2 weeks × 8 A100-class GPUs. Small labs win on interaction studies, failure boundaries, accounting — not on the big training race. |

## Convergent constraints (4/4)

1. ROI delta-QP baseline with identical masks is **the thesis test**, not a courtesy baseline.
2. GOP-matched differentiable proxy required for any video training (NHK: mismatch flips gains to losses).
3. Gain window may be narrow (mid QPs only) — per-QP tables mandatory.
4. Diffusion-based temporal editing cut (infeasible cost, no differentiable path, no metrics).
5. Evaluation floor: ceiling-normalized mAP, per-sequence distributions, ≥2 unseen detectors, ≥1 non-VTM codec, no GT-mask leakage, pre-registered thresholds.
6. Bar for the eventual video paper: ≥8–10% BD-rate vs anchor AND ≥3% BD-rate for the temporal module vs per-frame.

## Consequence for this repository

E1 (this repo) IS the fatal-1 test, made into a zero-training experiment with a mechanical go/no-go verdict. Every result — including a null — is publishable, and every later stage is gated on it. Full verdict text preserved in the session ARA report (workspace, outside this repo).
