# scope — bounded pre-editing for Video Coding for Machines, and the baselines that test it

Research harness for a pre-registered question:

> **Does bounded content-aware pre-editing (SCOPE-style) buy rate–accuracy gains that lossless in-codec ROI delta-QP cannot — at a ceiling-normalized task metric?**

Everything here is stage-gated and pre-registered **before** results exist:

| Doc | Content |
|---|---|
| [docs/MATH.md](docs/MATH.md) | Full mathematical model: bounded pre-edit operator, ceiling-normalized accuracy, PCHIP BD, gates G-η/G-1/G-2/G-t, falsification criteria F1–F5 |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | E1 (decisive, zero-training) + E2 (additivity) + E2t (temporal, gated) protocols, threat register |
| [docs/COUNCIL.md](docs/COUNCIL.md) | 4-reviewer adversarial council verdict that shaped the design |

## Why this design

A 4-reviewer council (2026-11-07) found two fatal flaws in the naive "SCOPE → video" plan:
1. **VTM ROI delta-QP** (MPEG EE m64712) is a lossless competitor nobody in the pre-editing literature has beaten or even compared against — it *is* the thesis test (arm A1).
2. **BD-rate is ill-defined when the method edits the signal itself** — every arm gets its own uncompressed ceiling $m^*$ and all comparisons run on ceiling-normalized quality (arm A2/A3 include $\eta$, the ceiling drop, as a first-class result).

Both fixes are built in, not bolted on. The experiment is **zero-training**: either outcome — pre-editing survives gate G-1, or it doesn't — is a publishable result.

## Quickstart

```bash
pip install -r requirements.txt       # core: numpy scipy pandas pyyaml pillow pytest
python scripts/check_encoders.py      # smoke-test ffmpeg/x265 toolchain + ROI map — run first
python -m pytest tests/ -q            # BD sanity: identical curves = 0, halved rate = -50%
cp configs/e1_decisive.yaml my_e1.yaml   # fill in data paths (>=8 sequences)
python scripts/make_masks.py my_e1.yaml  # shared per-sequence masks (one file feeds every arm)
python scripts/run_e1.py my_e1.yaml   # encode/pre-edit grid (resumable)
python scripts/eval_e1.py my_e1.yaml --weights <scorer.pt>   # quality.csv + ceiling.csv
python scripts/report_e1.py <workdir> # BD tables + gates + GO/NO-GO verdict
```

## Arms (E1)

| Arm | What it tests |
|---|---|
| A0 anchor | codec on original video |
| A1 roi | codec + per-LCU background QP offsets from the **same masks** — the lossless competitor |
| A2 preedit_fb | training-free blur-background pre-edit (α=β=0 member of the SCOPE family) |
| A3 preedit_scope | released [miccunifi/SCOPE](https://github.com/miccunifi/SCOPE) per-frame (enabled after weight integration) |
| A4 preedit+roi | interaction arm |

## Gates (pre-registered)

- **G-η**: ceiling drop η ≤ 0.01 per arm, else F2 fires (gains are accuracy recovery, not compression).
- **G-1**: median per-sequence BD gap (pre-edit − ROI-delta-QP) ≤ −1% **and** one-sided Wilcoxon p < 0.05 → premise alive; else F1 fires → pivot to the proxy-fidelity study, publish E1 as null + ROI baseline.
- E2 additivity κ and E2t temporal bar (+3% BD-rate, ID-switch reduction) are specified in MATH.md §7–8 and gated on E1.

## Status

- [x] Mathematical model + pre-registration (MATH/PROTOCOL/COUNCIL)
- [x] BD engine + tests; ceiling normalization; gate statistics
- [x] E1 harness: pre-editors, mask caching, codec wrappers (x264/x265), resumable runner, report with verdict
- [ ] x265 `qp-delta-map` format verification against installed version (`scripts/check_encoders.py`)
- [ ] SCOPE weight integration (arm A3)
- [ ] Data: MOT17 subset + COCO-ized GT; frozen scorer mapping verification
- [ ] E2: VTM/NNVC-4.0 build + additivity matrix
- [ ] E2t: only after G-1 and G-2 pass

## Key prior art (verified full-text)

- SCOPE: Semantic COdec Pre-Editing (arXiv:2610.39778) — the bounded pre-edit operator, α=0.09/β=0.03
- Zhao et al., A Preprocessing Framework for Video Machine Vision under Compression (DCC, arXiv:2512.15331) — restoration-style temporal preprocessor; per-task models
- NHK GOP-Based Deep Preprocessing (PCS 2024) — GOP-matched proxy necessity
- Neural Wrapping (CVPR 2025) — pre-only failure mode (+17.9% SSIM BD)
- Różek et al. VCM (VCIP 2023 / arXiv:2502.00565) — RoI+tracking baseline space, per-sequence regression risk
- MPEG VCM EE m64712 (PUT RoI preprocessing) — the lossless baseline this project is built around

## License

MIT — see [LICENSE](LICENSE).
