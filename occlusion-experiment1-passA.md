# Extend landmark occlusion masking (Change 1) to Pass A

Companion to `occlusion-experiment1.md`. That experiment's Change 1
(`landmark_occlusion_masking`) drops individual 2D landmarks/closure terms
whose GT position falls in an occluded region of the frame, so hallucinated
MediaPipe/FAN "closed mouth" labels during hand-over-mouth occlusion stop
actively supervising the model toward "closed." It was deliberately scoped to
**Pass C only** in the original design, on the assumption Pass C would run
*alone* (`pass_pattern: [C]`) for that experiment - Pass A was explicitly
called out as "untouched... a follow-up experiment."

## Why this is needed now

The experiment was later re-run with `pass_pattern: [A, C]` (Pass A
alternating back in, to keep refreshing SViT/heads/UNet rather than freezing
them for the whole run). That reintroduces exactly the problem Change 1 was
built to fix: Pass A's own landmark/closure loss (`run_pass_a` ->
`compute_2d_reconstruction_losses`) is still **fully unmasked**, so every
other training step SViT gets re-taught "predict closed mouth when the mouth
is covered" directly from the hallucinated GT labels - confirmed from a real
training log
(`slurm_jobs/output/stage2_passC_occlusion_exp_5981910.out`, `pass=A` lines
show nonzero, actively-training `landmark`/`closure` loss throughout).
Architecturally this matters more than it might seem: TT's output is
`residual (SViT's own per-frame prediction) + delta`, so even a well-trained
TT delta has to *overpower* a continuously-reinforced SViT bias every other
step, not just refine a neutral baseline. After ~4000 steps (~2000 of them
Pass A), the closed-mouth-on-occlusion artifact was still visible in demo
output.

**Goal of this change:** apply the exact same per-landmark occlusion masking
mechanism (already implemented and tested for Pass C) to Pass A's landmark/
closure loss too, reusing the same `Stage2Config.landmark_occlusion_masking`
flag - not a new, separately-toggleable flag - so "masking on" means "masking
on everywhere it's wired up," matching the plain reading of the config name.

## What's already in place (don't reimplement)

- `model/losses/landmark.py`: `landmark_visibility_mask(face_mask,
  landmarks_norm)` and the `mask` parameter on `fan_boundary_loss`/
  `mediapipe_landmark_loss`/`eye_closure_loss`/`lip_closure_loss` - fully
  implemented, tested (`tests/test_landmark_occlusion.py`), and generic (not
  Pass-C-specific in any way).
- `training/stage2.py::_compute_2d_reconstruction_losses_from_encoded` already
  has the `landmark_occlusion_masking: bool = False` parameter and the masking
  logic (computes `fan_occlusion_mask`/`mp_occlusion_mask` via
  `landmark_visibility_mask` when `True`, feeds them into the landmark/closure
  `gated_loss` calls). **This function itself needs no changes** - it's shared
  by both Pass A and Pass C already; only its *callers* differ in whether they
  pass `landmark_occlusion_masking=True`.
- Pass A's batches already carry everything needed: `keys_2d_a` (in
  `train()`) already includes `"face_mask"` and `"landmarks_fan"`/
  `"landmarks_mp"` - no new data plumbing required, purely a matter of
  threading the existing boolean through.

## Changes required (all in `training/stage2.py`)

1. **`compute_2d_reconstruction_losses`** (Pass A's loss-computation
   function, currently ~line 335-347): add a `landmark_occlusion_masking:
   bool = False` parameter, forward it into its
   `_compute_2d_reconstruction_losses_from_encoded(...)` call (currently
   called with no `landmark_occlusion_masking`/`precomputed_flame_out`
   kwargs - just add the one new kwarg; `precomputed_flame_out` stays
   Pass-C-only, no FLAME-forward-once restructuring needed here). Update its
   docstring, which currently only describes the unmasked path.

2. **`run_pass_a`** (currently ~line 350+): add the same
   `landmark_occlusion_masking: bool = False` parameter, forward it into its
   `compute_2d_reconstruction_losses(...)` call.

3. **`train()`'s Pass A call site**: pass
   `landmark_occlusion_masking=cfg.landmark_occlusion_masking` into
   `run_pass_a(...)`, the same config field already used for Pass C's call
   site - do NOT add a second/separate config field. Both call sites end up
   reading the same `Stage2Config.landmark_occlusion_masking` value.

4. **`training/config.py`**: update `landmark_occlusion_masking`'s docstring
   - it currently says "never threaded into Pass A's own call site" (written
     when this was Pass-C-only); that line becomes wrong and needs rewriting
     to reflect that it now covers both Pass A and Pass C.

5. **`training/stage2.py` module docstring / `_compute_2d_reconstruction_losses_from_encoded`'s
   own docstring**: both currently say things like "Pass A's own call site
   never passes True, so its behavior is unchanged" - update this framing,
   since that's no longer true once this change lands.

## Explicitly NOT part of this change

- No change to Pass B (`run_pass_b`/`compute_cycle_losses`) - Pass B has no
  landmark loss at all (it's the expression-cycle/augmentation pass), so
  masking doesn't apply there.
- No change to `model/losses/landmark.py` itself - the masking primitives are
  already generic and don't need touching.
- No change to Change 2 (vertex-space smoothness) - unrelated to this fix.
- No new `Stage2Config` field - reuse `landmark_occlusion_masking` as-is (see
  "why this is needed now" above for the reasoning).

## Verification

1. Run the existing test suite (`pytest tests/`) - nothing here should break
   any existing test, since every new parameter defaults to `False`/matches
   current behavior for any caller that doesn't pass it.
2. Quick synthetic smoke test (mirrors the one already used to validate
   `run_pass_c`'s FLAME-forward-once restructure): construct a tiny batch,
   call `run_pass_a(..., landmark_occlusion_masking=True, ...)` directly with
   real model modules, confirm it runs without shape errors and produces
   finite `landmark`/`closure` metrics.
3. **Operational note**: this is a source-code change, not a config value -
   the two currently-running jobs
   (`stage2_passC_occlusion_exp`/`_both_smooth`) were started with the OLD
   code (Pass A unmasked) and won't pick this up until killed and resubmitted
   against the updated code. Decide whether to let them finish as a
   (Pass-A-still-unmasked) baseline first, or restart them once this lands.
4. Real check: resume training from a recent checkpoint with this change
   live, watch Pass A's `landmark`/`closure` metrics (should behave the same
   way Pass C's already do - occasionally near-zero on heavily-occluded
   batches, not just uniformly reduced), and re-check a demo video after
   enough steps to see whether the closed-mouth-on-occlusion artifact
   actually softens now that Pass A is no longer working against Pass C's
   fix every other step.
