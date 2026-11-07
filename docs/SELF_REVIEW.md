# Self-Review — 2026-11-07 (post-push adversarial pass)

Third critique layer: after the 4-reviewer council and 2 critique subagents, a full
self-review traced every module again. Found **3 real bugs the earlier passes missed**,
plus gaps. All critical items fixed and smoke-tested before this file was committed.

## Bugs found and fixed

| # | Severity | Where | Bug | Fix |
|---|---|---|---|---|
| 1 | major | `bd.py::_clean` | Rate-dedupe reordered rows by rate → quality axis could come back **descending** → `bd_rate` silently returned NaN on legitimate duplicate-rate RD data (regression test added) | Re-sort by quality after rate-dedupe; `bd_quality` sorts by rate locally; both covered by tests |
| 2 | major | `experiment.py` | The "streaming" fix was incomplete: `write_qp_delta_map([masks[t] for t in range(n)])` still materialized the whole mask sequence (~50 GB at MOT17 scale) | `write_qp_delta_map` now accepts LazyMasks + `n_frames`, pulls per frame |
| 3 | major | pipeline | No script produced `masks/*.npz`, `quality.csv`, `ceiling.csv` — the README flow was broken in the middle (the one code-review item skipped in the previous pass) | Added `scripts/make_masks.py` (shared-mask caching) and `scripts/eval_e1.py` (quality + symmetric lossless ceilings, resumable); README quickstart updated |
| 4 | minor | `experiment.py` | A2c blur-everything control arm existed in docs but not in code; `np` missing (NameError on first run — caught by own smoke test); fps hardcoded 30 | A2c implemented; import fixed; `fps` moved to config |
| 5 | minor | `eval_e1.py` (first draft) | Ceiling dedup check built a reversed-tuple set — always-miss | Simplified to `(sid, arm) in c_done` |

## Verified working (integration, real ffmpeg)

- A0 / A2 / A2c arms encode end-to-end on synthetic data; rates differ plausibly
  (anchor 1642 → selective pre-edit 105 kbps at QP27 on a noisy synthetic).
- A1 on stock x265 fails **loudly** (`Unknown option: qp-delta-map`) — the thesis
  baseline cannot silently degenerate into A0. Requires a map-capable encoder build.
- Gate statistics: strong-gain case passes (p=0.0039 @ n=8), null fails, NaN gaps
  hard-fail with reason.

## Known open items (not bugs, declared)

1. **A1 needs an encoder with verified per-block QP offsets** (x265-VCM/vendor build,
   VTM per-CTU path, or VVenC ROI) — `check_encoders.py` is the gate; map format and
   sign remain VERIFY until a real build passes the bitrate-ordering self-test.
2. SCOPE weights integration (arm A3) — adapter marked VERIFY against upstream API.
3. Per-domain medians (MOT vs COCO) — metric exists in review; report groups only by
   sequence until a `domain` field is added to the config.
4. Skip-run / coefficient-count decoder statistics — E2 deliverable (mechanism claim
   deferred per MATH.md §3 caveat).
5. **Repo is public** — the pre-registered research plan is visible; flip to private
   if scoop risk (R4: ByteDance window 6–12 months) worries the owner.
6. Determinism: encoder version strings land in the run manifest, but cross-machine
   bit-exactness is not claimed.

## Process note

The two subagent critiques (math, code) each found real issues, and the self-review
still found a third layer — including one item the code reviewer explicitly listed
that was skipped during the first fix pass (missing eval scripts). Lesson recorded:
fix-lists get exhausted item-by-item or they silently rot.
