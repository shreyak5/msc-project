# Stage 2 config reference

Field-by-field reference for `training/config.py`'s `Stage2Config` (Stage 2's
YAML config, loaded by `load_stage2_config`), plus a short summary of the
`PretrainConfig`/`EvalConfig` dataclasses used alongside it. Every field below
is documented inline in `training/config.py` itself - this is a scannable
index into that, not a replacement for it. See `occlusion-handling.md`,
`occlusion-experiment1.md`, and `occlusion-experiment1-passA.md` for the
design write-ups several of the Pass C occlusion-experiment fields below come
from.

**Convention**: every field added after `Stage2Config`'s original core set
defaults to a value that reproduces the exact previous behavior - so any
existing YAML that doesn't mention a field is completely unaffected by that
field's existence. This is true throughout the table below; it's called out
again per-field only where the "safe default" isn't just `False`/`0`.

## Core training loop

| Field | Type | Default | What it does |
|---|---|---|---|
| `seed` | int | required | `torch.manual_seed` at startup. |
| `learning_rate` | float | required | One shared LR for the combined svit+heads+unet+tt Adam optimizer - each `run_pass_*` gates which subset actually moves via its own `requires_grad_` toggling. |
| `num_steps` | int | required | Total training steps. |
| `log_interval_steps` | int | required | How often (in steps) to print/log metrics. |
| `checkpoint_interval_steps` | int | required | How often to save a checkpoint. |
| `device` | str | required | `"cuda"` or `"cpu"`. |
| `checkpoint_dir` | str | required | Stage 2's own checkpoint directory - separate from Stage 1's. |
| `stage1_checkpoint_pth` | str | required | One-time seed (svit+heads only, no optimizer/unet/tt) loaded at startup, only when there's no Stage-2-own checkpoint to resume from (`checkpoint_pth` unset). Ignored on a resume. |
| `dataloader_config_path` | str | required | Path to `dataset_processing/config/dataloader.yaml`-shaped config. |
| `datasets_yaml_path` | str | `DEFAULT_DATASETS_YAML` | Which dataset registry YAML to use (e.g. `dataset_processing/config/datasets_sign_language_only.yaml` restricts Pass C's `2d_video` batches to sign-language datasets only). |
| `num_expression_params` | int | `100` | FLAME expression coefficient count. Must match `stage1_checkpoint_pth`'s (and, on resume, `checkpoint_pth`'s) own dim - validated on load, not silently mismatched. |
| `checkpoint_pth` | str \| None | `None` | Resume Stage 2's own training from this file - checked every run, unlike `stage1_checkpoint_pth`. `None` = fresh run. |
| `resume_optimizer_and_step` | bool | `True` | With `checkpoint_pth` set: `True` resumes optimizer state + step count too; `False` just seeds weights and starts fresh at step 0 with a new optimizer (e.g. after a `tt_variant` change, or seeding from a `freeze_encoder`-phase checkpoint whose optimizer momentum is stale for the next phase). |
| `freeze_encoder` | bool | `False` | Keeps svit/heads frozen in Pass A regardless of that pass's own default-unfreeze behavior - a UNet-warmup phase knob. Only meaningful when `pass_pattern` excludes `"B"`/`"C"` (both would still move the encoder). |

## Pass pattern

| Field | Type | Default | What it does |
|---|---|---|---|
| `pass_pattern` | list[str] | `["A", "B", "C"]` | Which passes to round-robin through, one per step (`step % len(pass_pattern)`) - e.g. `[A]` for encoder-only, `[A, C]` to alternate encoder training with TT training (skipping the cycle-augmentation Pass B), `[C]` to spend every step on TT alone. |
| `pass_b_encoder_steps` | int | `1` | Pass B cycles through 3 modes across *consecutive Pass B calls* (not outer-loop steps): this many calls update (tokens+SViT+heads) only. |
| `pass_b_unet_steps` | int | `1` | ...then this many calls update UNet only. |
| `pass_b_joint_steps` | int | `0` | ...then this many calls update both together, before repeating. Defaults `(1, 1, 0)` reproduce the original SMIRK-style pure alternation. `load_stage2_config` raises if all three are 0 while `"B"` is in `pass_pattern`. |
| `warmup_steps` | int | `1000` | Linear LR ramp from `learning_rate/warmup_steps` up to `learning_rate` over the first `warmup_steps` steps, then flat. `0` disables warmup (flat LR throughout). Added after Stage 2's fresh Adam optimizer + several previously-nonexistent losses caused a ~100x landmark-loss spike within 150 steps of Stage 2 starting. |

## Eval

| Field | Type | Default | What it does |
|---|---|---|---|
| `eval` | `EvalConfig` | see below | Periodic dev-set eval during training (`run_periodic_eval_local`/`aggregate_and_print_eval_results`) - reports landmark + temporal-smoothness scores per dataset. |

`EvalConfig` fields: `interval_steps` (int, default `500`, `0` disables periodic eval entirely), `num_clips_per_dataset` (int, default `256` - **total** across all ranks, not per-rank), `datasets` (list[str], default `["how2sign", "phoenix2014t", "csl_daily"]`).

## Pass C: synthetic occlusion

See `occlusion-handling.md` for the full design rationale.

| Field | Type | Default | What it does |
|---|---|---|---|
| `pass_c_synthetic_occlusion_enabled` | bool | `False` | While enabled, some real, visible frames in each Pass C clip are pretended-occluded for TT's *input* only (routed through `_fill_missing_frame_tokens` neighbor-averaging, visibility zeroed), while still supervising against that frame's real, uncorrupted target - teaches TT to recover a frame's parameters from temporal context alone. The two fields below are ignored unless this is `True`. |
| `pass_c_synthetic_occlusion_prob` | float | `0.1` | Per-eligible-frame independent probability of being chosen as synthetically occluded (no multi-frame bursts). A starting guess (with `TT_WINDOW_SIZE=15`, gives ~77% of any frame's local attention window at least one occluded neighbor), not a calibrated value. |
| `pass_c_occlusion_loss_weight` | float | `2.0` | Multiplier on synthetically-occluded frames' contribution to Pass C's losses (2D reconstruction + temporal smoothness) - e.g. `2.0` = occluded frame contributes 2x a normal frame's loss on top of its normal 1x share. `1.0` is a no-op. |

## Pass C occlusion-experiment (`occlusion-experiment1.md` / `occlusion-experiment1-passA.md`)

| Field | Type | Default | What it does |
|---|---|---|---|
| `landmark_occlusion_masking` | bool | `False` | Change 1: drop (zero-weight) individual landmarks whose GT 2D position falls in the occluded region of the frame, plus matching per-pair masking for eye/lip closure terms. Threaded into both Pass A's and Pass C's loss call sites. `False` reproduces the exact original unmasked landmark/closure loss everywhere. |
| `expressive_region_smooth_weight` | float | `3.0` | Change 2: peak weight for the gated expressive vertex regions (lips, eye_region, left/right_eye_region, nose, forehead) under full occlusion (`gate=1`). Inert whenever `temporal_vertex_smoothness_weight` is `0`, regardless of this field's own value. |
| `temporal_vertex_smoothness_weight` | float | `0.0` | Change 2: overall weight of the vertex-space, region-weighted, visibility-gated temporal smoothness term (`model/losses/temporal_smoothness.py`'s `vertex_velocity_penalty`). `0.0` fully disables the term - no separate "enabled" boolean exists. |
| `param_smoothness_in_pass_c` | bool | `True` | Gates the *existing* param-space velocity term (expression/jaw/camera/shape) on/off in Pass C. `True` reproduces the original always-on behavior. |
| `mouth_gate_use_region_visibility` | bool | `False` | Change 2 refinement: use a mouth-region-specific visibility (fraction of unoccluded lip landmark points) instead of the whole-face `visibility_ratio` score as the signal feeding the vertex-space gate below. `False` uses the whole-face score. |
| `pass_c_identity_pooling` | bool | `False` | Replaces `encode_video`'s per-frame decoded `shape` (FLAME identity) with a per-clip masked mean, broadcast back to every real frame (`model/encoding.py`'s `_pool_identity`) - identity becomes architecturally constant within a clip, rather than only softly discouraged from drifting via the param-space velocity term above. Unweighted by visibility (the working hypothesis is that TT's own attention already accounts for visibility, so re-weighting again here would be redundant). Also threaded into `run_periodic_eval_local`'s own `encode_video` call, so eval reflects the same behavior training optimizes. `False` reproduces the original per-frame shape behavior exactly. |
| `vertex_gate_mode` | str | `"min_vis"` | Which formula computes Change 2's vertex-space gate (`model/losses/temporal_smoothness.py`'s `compute_vertex_gate`). `"min_vis"` (`gate = 1 - min(vis[t], vis[t+1])`) reproduces the original formula exactly, but is poorly calibrated against this project's real `visibility_ratio` data - it never sits near 1.0 even on clean frames (median 0.68–0.78 measured across csl_daily/how2sign/phoenix2014t), so the gate rarely nears its floor and isn't very selective. `"delta_vis"` gates on the frame-to-frame *change* in visibility instead - see the next two fields. |
| `vertex_gate_delta_cap` | float | `0.1` | `"delta_vis"` mode only (ignored otherwise): normalizes the raw frame-to-frame `\|vis[t] - vis[t+1]\|` diff against this cap before clamping to `[0, 1]` - a diff at/above `cap` saturates to gate `1.0`. `0.1` is roughly the global p95 frame-to-frame visibility diff measured across csl_daily/how2sign/phoenix2014t's cached `visibility_ratio` (per-dataset medians ranged 0.007–0.024, p99 0.066–0.223). |
| `vertex_gate_delta_beta` | float | `1.0` | `"delta_vis"` mode only (ignored otherwise): `gate = norm ** beta`, where `norm` is the cap-normalized diff above. `beta=1` is a no-op (`gate=norm`, plain linear). `beta > 1` suppresses small/moderate `norm` values much faster than large ones (a value already near 1 barely changes under any power) - widening the separation between genuine transitions and ordinary jitter as `beta` increases, i.e. higher `beta` means higher-diff frames dominate *more* over low-diff ones. Named `beta`, not `gamma`, specifically to avoid confusion with `tt_gamma` below, which behaves oppositely (`x ** gamma`, not `x ** (1/gamma)`-style boosting) on a different signal entirely. Measured against real data: at `beta=1`, median gate is already fairly low (0.07–0.24, dataset-dependent); at `beta=1.5` (this experiment's own starting value), the median drops another 2–3x while the fraction of pairs above `0.8` barely moves - i.e. `1.5` sharpens selectivity without meaningfully weakening the genuine-transition tail. |

## TT architecture

| Field | Type | Default | What it does |
|---|---|---|---|
| `tt_variant` | str | `"original"` | Which `model/temporal.py` class to use: `"original"` (`TemporalTransformer` - 7 QK+ALiBi heads + 1 dedicated visibility-only head), `"simple"` (`SimpleTemporalTransformer` - all 8 heads QK+ALiBi, no visibility input at all), or `"gated"` (`GatedTemporalTransformer` - all 8 heads QK+ALiBi with a uniform post-softmax visibility gate + renormalize). Changing away from `"original"` means a `checkpoint_pth`'s saved `tt` weights won't shape-match - `train()` skips loading `"tt"` in that case (svit/heads/unet still warm-start; `tt` starts fresh). |
| `tt_gamma` | float | `constants.TT_GATE_GAMMA` (`3.0`) | `GatedTemporalTransformer`'s post-softmax visibility-gating exponent (`gate_j = clamp(vis_j, 0, 1) ** gamma`) - reshapes each key frame's *raw* visibility inside TT's own attention. Ignored unless `tt_variant == "gated"`. Not the same knob as `vertex_gate_delta_beta` above - different formula direction, different signal, different consumer (attention vs. loss weighting). |

## wandb

| Field | Type | Default | What it does |
|---|---|---|---|
| `wandb_project` | str | `"msc-project-stage2"` | wandb project name. |
| `wandb_entity` | str \| None | `None` | wandb entity/team. |
| `wandb_run_name` | str \| None | `None` | wandb run name - `None` lets wandb auto-generate one. |

## `PretrainConfig` (Stage 1) and `EvalConfig`, briefly

`PretrainConfig` is Stage 1's equivalent config (`load_pretrain_config`) - a subset of the same shape: `seed`, `learning_rate`, `num_steps`, `log_interval_steps`, `checkpoint_interval_steps`, `device`, `checkpoint_dir`, `dataloader_config_path`, `datasets_yaml_path`, `num_expression_params`, `checkpoint_pth`, and the same three `wandb_*` fields. No pass-pattern/occlusion/TT fields - Stage 1 only ever trains svit+heads on single images, no TT involved at all.

`EvalConfig` is Stage 2's nested periodic-eval config - see the **Eval** section above.

## Naming note: the two `passC_*` experiment families

`stage2_passC_occlusion_exp*.yaml` (original, pre-existing) and `stage2_passC_id_pooling*.yaml` (added alongside `pass_c_identity_pooling`/`vertex_gate_*`) are deliberately separate experiment lines with distinct `checkpoint_dir`/`wandb_run_name` values, so they never collide in directory listings, wandb, or SLURM job names. `stage2_passC_id_pooling.yaml` isolates `pass_c_identity_pooling` as the only new variable against the existing gated-TT baseline; `stage2_passC_id_pooling_gate_reshape.yaml` adds `vertex_gate_mode: delta_vis` on top of that.

