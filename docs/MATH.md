# Mathematical Model — Pre-Editing for Video Coding for Machines (VCM)

**Project:** `wagur1/scope` — Bounded content-aware pre-editing and its decisive baselines under VCM evaluation.
**Status:** v1.0 (2026-11-07), incorporating the 4-reviewer council verdict (see `COUNCIL.md`).
**Direction (finalized):** Stage-gated program. **E1 (decisive, zero-training)** answers: *does bounded pixel-domain pre-editing buy anything that lossless in-codec ROI delta-QP cannot?* E2 (additivity + temporal, gated on E1) follows only if the premise survives. This document fixes the mathematics both stages share.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $V = (I_1, \dots, I_T)$ | input video, $I_t \in [0,1]^{3\times H\times W}$ |
| $\mathcal{C}_{f,s}$ | standard codec, family $f$ (VVC/HEVC/AVC), operating point $s$ (QP) |
| $R(\cdot)$ | encoded bitrate (kbps) of the argument sequence |
| $\mathcal{D}$ | frozen downstream task model (detector / tracker) |
| $\mathrm{mAP}_\mathcal{D}(V')$ | task accuracy of $\mathcal{D}$ on decoded sequence $V'$ |
| $E_\theta(\cdot\,;\,c)$ | pre-edit operator, conditioned on $c=(f,s)$ |
| $M_t \in [0,1]^{1\times H\times W}$ | task-relevant mask (predicted at test time; GT never used at inference) |

## 2. The pre-edit operator

Bounded decomposition (SCOPE form, verified in arXiv:2610.39778):

$$
\tilde I_t \;=\; A_t \odot (I_t + \Delta^{\mathrm{obj}}_t) \;+\; (1-A_t)\odot\big(\mathrm{Blur}_{k}(I_t) + \Delta^{\mathrm{bg}}_t\big),
$$

with constraints

$$
A_t \in [0,1]^{1\times H\times W},\qquad
\|\Delta^{\mathrm{obj}}_t\|_\infty \le \alpha,\qquad
\|\Delta^{\mathrm{bg}}_t\|_\infty \le \beta,\qquad
\mathrm{Blur}_k = G_{\sigma(k)} * I_t ,
$$

SCOPE uses $\alpha = 0.09$, $\beta = 0.03$ (both stated in the paper). The **bounds are load-bearing** (council R1-obj1): they cap content destruction so the edit remains "human-tolerable" and make the ceiling claim of §4 meaningful.

**Blur bound (pre-registered, math-review M2):** the blur kernel is part of the edit and must not be tuned post-hoc: $\sigma = 8.0$ px is fixed in this document before any E1 results. The training-free arm A2 additionally reports a **background fidelity floor** — PSNR of the edited background region vs the original — as a diagnostic; a *blur-everything* control arm (mask ≡ 0) isolates how much of the gain is content destruction rather than selectivity.

**Mask source.** At training time SCOPE uses ground-truth boxes; at inference $M_t$ is predicted by the pre-editor's own attention head. For E1 with the training-free fallback editor, $M_t$ comes from a public detector (YOLO-family). **The identical $M_t$ sequence must be reused by both the pre-editor and the ROI delta-QP baseline** (council R3-R1) — masks are computed once per sequence and cached. The detector checkpoint is pinned in the run manifest (math-review M4: an unpinned checkpoint is a forking path).

## 3. Baseline: ROI delta-QP (lossless, in-codec)

Given the same masks, the encoder applies a per-CTU/LCU quantization offset. All parameters are **pinned** (math-review M4):

$$
QP^{\mathrm{ROI}}_t(u) \;=\; QP \;+\; \delta\cdot \mathbb{1}\Big[\overline{M_t}(u) < \tau\Big],\qquad
\delta \in \{4,8,12,16\},\quad \tau = 0.5,
$$

where $\overline{M_t}(u)$ is the mean mask over CTU/LCU $u$ (majority rule, CTU 64). Rate control is fixed-QP (CQP) with **AQ and cutree disabled** on every arm — an ROI offset under adaptive quantization is not the §3 baseline. The map's sign and granularity are pinned **empirically** by `scripts/check_encoders.py` (bitrate-ordering self-test), never by convention. This is a pure rate-domain intervention: $I_t$ is never modified, and the decoded video reconstructs the original content.

**Mechanism hypothesis (why pixel edits could still win).** At fixed QP, background blocks under delta-QP still carry (attenuated) texture residuals and motion vectors; a background that is *actually flat* emits skip flags, zero-MV inter blocks, and near-empty coefficient sets. Formally the claim is

$$
R_{QP}\big(E_\theta(V)\big) \;<\; R_{QP}^{\mathrm{ROI}}(V)
\quad\text{at equal normalized task accuracy},
$$

**with two pre-registered caveats.** (i) The mechanism claim (skip/zero-MV) is *reported only if* decoder statistics support it — E1 records bitrate only; skip-run and coefficient-count logging is an E2 deliverable, so E1 asserts the *inequality*, not the mechanism (math-review M6). (ii) On moving-camera content (MOT17) a blurred background still carries global motion; the mechanism story must not be narrated for sequences where it cannot hold.

## 4. Ceiling-normalized accuracy (council R3-R2 fix)

Pre-editing changes the signal, so raw $\mathrm{mAP}$ of decoded streams is measured on *different content* per arm and BD comparisons are formally invalid. Fix — with the ceiling protocol made **symmetric** (math-review C2: an asymmetric ceiling conflates edit destruction with the light codec's own loss and biases the 1% gate):

$$
m^*_E \;=\; \mathrm{mAP}_\mathcal{D}\big(\mathrm{Dec}_{\ell}\big(E_\theta(V)\big)\big),
\qquad
m^*_{\mathrm{anchor}} \;=\; \mathrm{mAP}_\mathcal{D}\big(\mathrm{Dec}_{\ell}(V)\big),
$$

where $\mathrm{Dec}_{\ell}$ is the **identical** negligible-compression path for both: x265 lossless (`qp=0, lossless=1, yuv444p`). Both ceilings pass through the same codec; neither is measured raw. Pin the settings in the run manifest.

**Estimand (math-review C3, stated explicitly).** Normalized quality answers *"rate needed to preserve a fraction of the arm's own uncompressed task accuracy"*. Deployment cares about absolute accuracy, so E1 reports **both**: (a) the normalized BD (primary, per-arm ceilings below) and (b) a secondary BD on the **common absolute mAP interval** truncated at $\min(m^*_E, m^*_{\mathrm{anchor}})$.

**Normalization policy:** content-modifying arms ($A2, A3, A4$) divide by their **own** $m^*$; non-modifying arms ($A0, A1$) inherit $m^*_{\mathrm{anchor}}$ — re-measuring a second ceiling for arms that share the anchor's content is pure added variance. Note normalized quality may exceed 1 (mild compression can help a detector); "ceiling" is then not a hard ceiling — reported as is, never clipped.

**Ceiling drop** (content destruction paid up front, gate applies to the **thesis arm only** — math-review M5: gating on diagnostic arms can kill a true premise):

$$
\eta \;=\; 1 - \frac{m^*_E}{m^*_{\mathrm{anchor}}}.
$$

**Gate G-η:** $\max_v \eta_v \le 0.01$ over sequences for the thesis arm; otherwise F2 fires for that arm. For tracking, the accuracy analogue is defined on **IDF1 only** (MOTA can be negative — ratios on it are meaningless; math-review m1).

## 5. Bjøntegaard deltas with PCHIP (exact algorithm used in `scope_vcm/bd.py`)

Given anchor points $\{(q_i, r_i)\}_{i=1}^{n}$ and proposal points $\{(\tilde q_j, \tilde r_j)\}_{j=1}^{m}$, where $q$ is (normalized) quality and $r$ is rate:

1. **Sort** both clouds by $q$; **deduplicate** equal $q$ (keep mean $r$).
2. **Common interval** $[q_{\lo}, q_{\hi}] = [\max(q^{a}_{1}, q^{p}_{1}),\; \min(q^{a}_{n}, q^{p}_{m})]$. If empty → BD undefined, return `NaN` (never extrapolate).
3. Fit rate as a function of quality with **monotone PCHIP** on $(q,\ \log_2 r)$ for both arms (log-domain rate is the VCEG convention; PCHIP avoids Runge oscillation of cubic splines).
4. Evaluate both interpolants on a dense grid over $[q_{\lo}, q_{\hi}]$ and integrate the mean difference:

$$
\Delta \;=\; \frac{1}{q_{\hi}-q_{\lo}} \int_{q_{\lo}}^{q_{\hi}} \big(\log_2 R_{p}(q) - \log_2 R_{a}(q)\big)\, dq .
$$

5. **BD-rate** $= \left(2^{\Delta} - 1\right)\times 100\%$ (negative = proposal saves rate). **BD-quality** fits quality as a function of $\log_2 r$ (PCHIP) and reports the mean quality difference over the common $\log_2$-rate interval — the exact mirror of the BD-rate integration. Wording note (math-review m4): VCEG-M33 specifies a single cubic *polynomial* fit; PCHIP is the JVET-common monotone alternative chosen here to avoid spline oscillation — a deliberate, stated deviation, not a claim of matching M33.

Quality axis: $\widetilde m$ for detection (§4); for tracking use MOTA and IDF1 as separate quality axes (temporal-sensitive ones per council R2), each yielding its own BD.

**Per-sequence reporting is mandatory** (council R3; Różek observed per-sequence regressions up to +27.17%): aggregate BD never appears without the per-sequence vector and its worst case.

## 6. E1 — the decisive experiment

Arms, all sharing the same masks:

| Arm | Definition |
|---|---|
| A0 anchor | $\mathcal{C}_{f,s}(V)$ |
| A1 ROI delta-QP | $\mathcal{C}^{\mathrm{ROI}}_{f,s,\delta}(V)$, $\delta$ swept |
| A2 pre-edit (fallback) | $\mathcal{C}_{f,s}(E_{\mathrm{fb}}(V))$ — training-free blur-background editor ($\alpha=\beta=0$ boundary of SCOPE), **co-primary thesis arm (zero-training baseline)** |
| A3 pre-edit (SCOPE) | $\mathcal{C}_{f,s}(E_{\mathrm{scope}}(V))$ — released miccunifi/SCOPE weights, image model applied per frame; **co-primary thesis arm (learned bounded pre-editing, $\alpha=0.09, \beta=0.03$)** |
| A4 pre-edit + ROI | $E_\theta$ then ROI delta-QP on remaining background statistics (interaction arm, diagnostic only — never gates) |

QP grid $\{22, 24, 27, 29, 32, 34, 37, 39, 42\}$ (VCM CTC values + intermediate mid-QPs; math-review M7: the gain window may be narrow, ΔQP=5 cannot resolve it), low-delay config.

**Thesis statistic.** Per sequence $v$, with the **strongest-configured-competitor rule** (math-review C1: comparing against an arbitrary $\delta$ is the cheapest way for a false premise to pass), evaluated for $A \in \{\text{A2}, \text{A3}\}$:

$$
g_v(A) = \mathrm{BD}_v(\text{A0}, A), \qquad
g^{\mathrm{roi}}_v = \min_{\delta \in \{4,8,12,16\}} \mathrm{BD}_v\big(\text{A0}, \text{A1}(\delta)\big).
$$

**Decision rule (gate G-1):** bounded pre-editing survives iff $A \in \{\text{A2}, \text{A3}\}$ satisfies

$$
\mathrm{med}\{g_v(A) - g^{\mathrm{roi}}_v\} \;\le\; -1\%\ \text{BD-rate}
\quad\text{and}\quad
P(\text{one-sided Wilcoxon} \le 0.05)
\quad\text{and}\quad
\max_v \eta_v(A) \le 0.01 .
$$

**Falsification scope pin:** If A2 fails G-1 before A3 is integrated, only the training-free blur premise ($\alpha=\beta=0$) is falsified; full falsification F1 of learned bounded pre-editing requires A3 (`ScopePreEditor`) to also fail G-1.

**Sample-size and soundness pins** (math-review C1/M3): minimum $n = 8$ sequences (below it the exact one-sided Wilcoxon cannot reach $p<0.05$); $n \ge 10$ preferred; sequences with NaN BD (disjoint normalized-quality intervals) **hard-fail the gate** — dropping them is a data-dependent selection rule; per-domain medians (MOT vs COCO-ized) are reported alongside the pooled median since they are different populations; a sign test is reported as robustness (Wilcoxon symmetry is shaky for ratio-derived BD gaps); the median gap carries a bootstrap CI (BCa preferred at small $n$).

If G-1 fails on both A2 and A3 → **F1 fires: the pixel-domain pre-editing premise is dead in this regime; the program pivots to the proxy-fidelity study (GAP-10)** — the null result is itself the paper (with A1 as the contribution: a clean ROI-delta-QP baseline the VCM literature currently lacks under matched masks).

## 7. E2 — additivity with in-loop neural tools (gated on G-1 pass)

Complementarity index between an encoder-side tool $X$ (pre-edit) and a decoder/in-loop tool $Y$ (NNVC-4.0 NNLF, film-grain synthesis), with the sign convention **pinned** (math-review M1: BD $<0$ = saving, so the raw difference of BDs inverts the intuitive direction — the index is defined on absolute savings):

$$
\kappa \;=\; \big|\mathrm{BD}(\text{A0}, X \cup Y)\big| \;-\; \big|\mathrm{BD}(\text{A0}, X)\big| \;-\; \big|\mathrm{BD}(\text{A0}, Y)\big|,
$$

computed per sequence with bootstrap CIs. $\kappa > 0$ → **super-additive** (the combination saves more than the sum of individual savings — complementary tools); $\kappa < 0$ → **sub-additive** (the tools overlap/redundant). Caveat carried from §4: $X$ modifies content, so its BD lives on its own normalized axis; the three BDs entering $\kappa$ must additionally be reported on the common **absolute** mAP interval before any mechanism claim. **Gate G-2:** $\kappa \ge 1\%$ BD-rate (super-additive) or $\kappa \le -1\%$ (sub-additive) on the median → interaction is a finding (either direction is publishable per R4); otherwise report the null honestly.

## 8. E2t — temporal consistency (gated on E2; the eventual video paper)

Edit field $E_t = (A_t, \Delta^{\mathrm{obj}}_t, \Delta^{\mathrm{bg}}_t)$. Consistency of the *edit* (not the pixels) under optical-flow warp $W_{t-1\to t}$:

$$
\mathrm{Flick}(A) \;=\; \frac{1}{T-1}\sum_{t=2}^{T} d\big(A_t,\; W_{t-1\to t}(A_{t-1})\big),
\qquad
d(U, U') = \frac{\|U - U'\|_1}{HW} \in [0,1].
$$

Training arms (only after G-1 and G-2 pass, per council R2/R4):
1. **per-frame**: $E_\theta$ applied frame-wise, no temporal term (this is A3 above);
2. **+consistency**: add $\lambda_{\mathrm{cons}} \sum_t d(E_t, W(E_{t-1}))$ to training loss, GOP-matched differentiable proxy (council R1-obj2: NHK showed GOP mismatch flips gains to losses; the proxy must reproduce VTM low-delay hierarchical-B reference structure);
3. **restoration-style**: Zhao DCC-style residual preprocessor as the control arm.

**Training loss (E2t arm 2, pre-registered design).** Mirrors SCOPE's verified structure (bg + attention + task + regularization) with rate and consistency terms:

$$
L \;=\; \underbrace{L_D\big(V,\ \hat V\big)}_{\text{distortion}}
\;+\; \lambda_R\, L_R\big(\bar V\big)
\;+\; \lambda_{\mathrm{task}}\, L_{\mathrm{task}}
\;+\; \lambda_{\mathrm{cons}} \sum_{t=2}^{T} d\big(E_t,\ W_{t-1\to t}(E_{t-1})\big)
\;+\; \lambda_{\mathrm{reg}}\, L_{\mathrm{reg}},
$$

where $\bar V$ is the proxy-encoded edited video, $\hat V$ its proxy reconstruction, $L_R$ the proxy rate (factorized-prior entropy model), $L_{\mathrm{task}}$ the frozen-task feature/accuracy loss, $d$ the edit-field distance of §8, and $L_{\mathrm{reg}}$ the $\ell_1$+TV bounds keeping $\|\Delta^{\mathrm{obj}}\|_\infty \le \alpha$, $\|\Delta^{\mathrm{bg}}\|_\infty \le \beta$ (hard-clamped, not just penalized). Reference points verified from the literature: SCOPE uses $L = L_{bg} + L_{att} + L_{task} + L_{reg}$ (masked-L1-to-blur + rate + BCE/Dice attention + frozen-detector Smooth-L1 feature matching + $\ell_1$/TV); Zhao DCC uses $L = 10\,(L_D + 0.001\,L_R) + L_{Acc}$ through an intra+inter virtual codec. **Standing warning (council):** Neural Wrapping showed pre-only training fails (+17.9% SSIM BD) — this loss trains the editor with a post-processor in the loop, or accepts iterative refinement against domain shift; proxy-vs-VTM rate error must stay < 10% on held-out clips.

**Gate G-t:** arm 2 must beat arm 1 by $\ge 3$ percentage points of BD-rate saving *and* reduce ID-switches (JDE) on tracking data *and* deliver $\ge 8\%$ BD-rate vs the A0 anchor over the same protocol (council constraint 6 / math-review m3: the anchor bar is part of the pre-registration, not an afterthought); otherwise the temporal module is not a contribution (council R2-obj1) and the video paper is not written.

## 9. Falsification criteria (pre-registered)

| ID | Claim | Falsified when | Consequence |
|---|---|---|---|
| F1 | Pixel pre-editing beats lossless ROI delta-QP | G-1 fails | Pivot to proxy-fidelity study; publish null + ROI baseline |
| F2 | Gains are compression, not accuracy recovery | $\eta > 0.01$ for the thesis arm (single condition, MATH §4) | No compression claim permitted; reframe as accuracy-preservation study |
| F3 | Interaction with in-loop NN tools exists | $|\kappa| < 1\%$ | Publish null as short paper; encoder-vs-decoder-NN debate stays open |
| F4 | Temporal consistency module adds value | G-t fails | No video paper; E1/E2 results stand alone |
| F5 | Training-free fallback ≈ learned pre-editor | TOST equivalence test vs pre-registered margin (1% BD-rate) fails to establish equivalence | Report difference as is — absence of significance is not equivalence (math-review m2) |

## 10. Scope of validity

- E1 conclusions hold for: x264/x265-style inter coding at QP 22–42, detection/tracking tasks with public anchors, bounded ($\alpha, \beta$) edits, predicted masks of the same source for all arms.
- E1 does **not** establish: VVC hierarchical-B behavior (needs the VTM arm of E2), one-model-many-codecs claims (needs ≥2 codec families — council R2: family conditioning with a single anchor is vacuous), human-perception claims (needs a formal subjective test — out of scope, reported as limitation).
- All thresholds ($1\%$, $3\%$, $\eta \le 0.01$, $\alpha=0.05$) are pre-registered here and in `PROTOCOL.md`; changing them after seeing results is prohibited by the project's own rigor rules (ARA D2).
