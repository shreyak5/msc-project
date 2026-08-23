# Implementation Plan: Pass C Occlusion Handling (Region-Masked Landmarks + Visibility-Gated Mouth Smoothness)

## Context

Current failure mode: on sign language video, hand-over-mouth occlusions cause the model to snap to a closed/neutral mouth. Root causes: (a) MediaPipe/FAN hallucinate plausible closed-mouth landmarks on occluded mouths, actively supervising "closed"; (b) with no reliable mouth signal, the expression/jaw L2 regularizer pulls toward neutral.

This change set targets **Pass C only** (temporal pass; TT is the only trainable component there — SViT, tokens, MLP heads, UNet all remain frozen). Passes A and B are untouched. A follow-up experiment (out of scope here) will apply landmark masking to Pass A.

Experiment design rationale: since only TT updates in Pass C, this run cleanly measures how much the temporal model alone can fix, before touching SViT.

## Change 1 — Occlusion-region-masked landmark loss (Pass C only)

Drop (zero-weight) individual landmarks whose 2D position falls inside the occluded region of the frame. Do **not** use frame-level score-proportional down-weighting — a hand-over-mouth frame can still score ~0.7 visibility, which would let hallucinated mouth landmarks through at 0.7 weight while needlessly down-weighting good brow/eye landmarks.

### Implementation

- Landmark loss becomes a per-landmark weighted sum: weight 0 if the landmark lies in the occluded region, 1 otherwise. Normalize by the number of *kept* landmarks, not total, so occluded frames don't silently shrink the loss magnitude.
- The eye-closure and lip-closure terms (SMIRK/DECA-style) must respect the same mask: if either landmark of a closure pair is occluded, drop that closure term for that frame.
- This masking applies to the landmark loss **in Pass C only**. Pass A/B landmark losses are unchanged in this experiment.
- Edge case: if all landmarks in a frame are occluded, the frame contributes zero landmark loss (guard against 0/0 in the normalization).

## Change 2 — Region-weighted, visibility-gated temporal smoothness on mesh vertices (Pass C only)

Replace/extend the current parameter-space velocity smoothness with a **vertex-space** velocity penalty that is (a) region-weighted with heavy weight on the mouth, and (b) **gated by per-frame visibility** so it only bites during occlusion.

### Why gating is required, not optional

Real mouthings are abrupt (plosives open/close the mouth in 1–2 frames at 25fps). An unconditional heavy mouth-smoothness penalty cannot distinguish "spurious snap-shut under occlusion" from a genuine /p/ /b/ /m/, and this project has already observed a fixed-mesh collapse from over-weighted smoothness once (see Sec 6 of the project plan: velocity weight lowered 1.0 → 0.1). Occluded frames have no legitimate reason for abrupt mouth motion; visible frames do. Therefore: heavy penalty only when occluded, light/zero when visible.

### Implementation

- Compute predicted FLAME vertices per frame (already needed for mesh loss; reuse).
- Velocity term: L1 on per-vertex first differences, `|V(t) − V(t+1)|`, mean-reduced with per-vertex region weights.
- Region weights from `FLAME_masks.pkl` (same source as the existing region-weighted mesh loss):
  - `lips`: high weight — config knob `mouth_smooth_weight`, starting value 3.0
  - other face regions covered by the existing mesh-loss weighting (`eye_region`, `nose`, `forehead`, `face`, boundary): 1.0
  - `left/right_eyeball` and non-face regions (neck, ears, scalp): 0.0, consistent with the existing mesh loss exclusions
- Visibility gating: scale the *mouth-region* portion of the penalty per frame-pair by an occlusion factor derived from the visibility scores of frames t and t+1:
  - `gate(t) = 1 − min(vis(t), vis(t+1))` — i.e., full mouth penalty only when at least one frame of the pair is fully occluded, tapering to 0 on clean pairs.
  - Effective mouth weight per pair: `1.0 + (mouth_smooth_weight − 1.0) * gate(t)` — so on fully visible pairs the mouth region falls back to baseline weight 1.0, never below the rest of the face.
  - Optional refinement (config-flagged, default off for v1): use a mouth-region-specific visibility (fraction of the mouth bounding region not covered by the occlusion mask from Change 1) instead of the whole-face score, since a hand at the forehead shouldn't trigger mouth smoothing. Implement the hook but keep whole-face score as default to limit moving parts in the first run.
- Non-mouth regions keep the penalty ungated at weight 1.0 (mild global smoothness, same spirit as the existing velocity term).
- The existing parameter-space velocity smoothness term (weight 0.1) — keep it implemented in the code, but **disable it in Pass C for this experiment** (config-flag it off, don't delete): the new vertex-space term is the *only* temporal smoothness active in Pass C. This keeps the ablation clean — one smoothness mechanism, in mesh space, mouth/occlusion-targeted.
- Overall weight of the new vertex velocity term: config knob `vertex_smooth_weight`, starting value 0.1 (match the previous param-space weight; tune on sign language dev splits).

## Training procedure for this experiment

- **Do not re-run full three-pass Stage 2.** Since Pass C freezes everything except TT, and only Pass C changes: start from the current best Stage 2 checkpoint and run **Pass C alone** (or a schedule heavily biased to C) for a short run. TT is 3 layers against a fixed encoder; the snap-shut artifact should visibly soften within a fraction of a full training run if the mechanism works.
- Data: **sign language video datasets only** (How2Sign, CSL-Daily, PHOENIX-2014T — official train splits). MEAD/AFEW-VA and all other categories are excluded from this experiment's Pass C batches. Full clips (TT-training mode), clip length ≥ w = 15 as per the existing plan.
- All other Pass C losses (photometric/VGG for 2D video, regularization) unchanged.
- Checkpoint separately from the main Stage 2 checkpoints (new subdirectory under `$PROJECTDIR`, e.g. `passC_occlusion_exp/`), so the baseline remains available for A/B comparison.

## Config additions (all knobs, with defaults)

| Knob | Default | Meaning |
|---|---|---|
| `landmark_occlusion_masking` | true (this experiment) | enable Change 1 in Pass C |
| `mouth_smooth_weight` | 3.0 | peak mouth-region vertex velocity weight under full occlusion |
| `vertex_smooth_weight` | 0.1 | overall weight of the new vertex-space velocity term |
| `param_smoothness_in_pass_c` | false (this experiment) | enable/disable the existing parameter-space velocity term in Pass C (kept in code, off for this run) |
| `mouth_gate_use_region_visibility` | false | use mouth-region visibility instead of whole-face score for gating |

## Explicitly out of scope (do not implement now)

- Synthetic occlusions (hand-pasting, pseudo-GT supervision).
- Any change to Pass A or Pass B (Pass A landmark masking is the planned *next* experiment if this one shows progress).
- Changes to TT architecture, visibility head, window size, or the visibility-score pipeline.