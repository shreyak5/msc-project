"""Stage 1/Stage 2 training config (implementation-plan.md Sec 7), mirroring
dataset_processing/dataloading/config.py's dataclass + YAML-loader pattern
(which itself holds multiple related config dataclasses in one file)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML
from model import constants


@dataclasses.dataclass
class PretrainConfig:
    seed: int
    # TokenFace's own fine-tuning LR (1e-4), not SMIRK's from-scratch 5e-4 - we're
    # fine-tuning a FaRL-pretrained transformer (SViT), not training a ResNet
    # encoder from scratch, so a lower LR to avoid wrecking the pretrained
    # features is the better anchor to borrow here.
    learning_rate: float
    # Step-based (not epoch-based): "epoch" isn't a well-defined progress unit
    # once a loop can draw from a loader indefinitely (training/loss_utils.py's
    # next_batch restarts the loader on exhaustion rather than stopping), and
    # Stage 2 needs steps regardless (it draws from two independently-cycling
    # loaders at different relative rates, so no single "epoch" spans both) -
    # Stage 1 matches for consistency between the two stages.
    #
    # 60000 starts from SMIRK's own reference schedule (60k iterations), but
    # that number was tuned together with SMIRK's own lr=5e-4 for from-scratch
    # training - we use a different, gentler lr (TokenFace's 1e-4, since we're
    # fine-tuning a pretrained transformer, not training from scratch), and LR
    # and step-count interact (total "distance traveled" in weight-space
    # depends on both together) in ways that pull in opposite directions here: a gentler
    # LR suggests possibly needing MORE steps, while fine-tuning from an
    # already-good initialization typically needs FEWER steps than from-scratch
    # training. Neither effect is derivable without empirical tuning, so this
    # is a starting guess, not a calibrated value - revisit empirically.
    num_steps: int
    log_interval_steps: int
    checkpoint_interval_steps: int
    device: str
    checkpoint_dir: str
    dataloader_config_path: str
    datasets_yaml_path: str = str(DEFAULT_DATASETS_YAML)
    # Number of FLAME expression coefficients the model is trained with (Sec 2.2
    # defaults to 100; SMIRK's own FaMoS expression templates, used by Stage 2
    # Pass B's cycle-consistency template injection, are natively only 50-dim -
    # see model/constants.py's EXPRESSION_TEMPLATE_NUM_DIMS). Sizes the expression
    # head's nn.Linear (model/heads.py's ComponentHeads) and FLAME's own PCA
    # basis slicing (model/flame/flame.py's n_exp) together - see training/
    # checkpoint.py for how a checkpoint's own recorded dim is validated against
    # this on load, since a mismatch isn't safely resumable (different-width
    # expression head weights). 100 (default) reproduces the original,
    # pre-configurable behavior exactly for every existing YAML.
    num_expression_params: int = 100
    # Resume training from this checkpoint file - checked on every run. None
    # (unset in the YAML) means a fresh run.
    checkpoint_pth: str | None = None
    # wandb (training/wandb_utils.py) run identity - all optional so existing
    # YAMLs need no changes. wandb_run_name=None lets wandb auto-generate a name.
    wandb_project: str = "msc-project-pretrain"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    cfg = PretrainConfig(**raw)
    if not 50 <= cfg.num_expression_params <= 100:
        raise ValueError(f"num_expression_params must be between 50 and 100, got {cfg.num_expression_params}")
    return cfg


@dataclasses.dataclass
class EvalConfig:
    """Periodic dev-set eval during Stage 2 training (training/stage2.py's
    run_periodic_eval_local/aggregate_and_print_eval_results) - reports a
    landmark score and a temporal-smoothness score per dataset, every
    interval_steps, against a fixed dev-split clip subset of `datasets`."""
    # 0 disables periodic eval entirely (no eval loaders built, no eval step
    # ever runs) - analogous to checkpoint_interval_steps/log_interval_steps,
    # which are always assumed positive; this field is the one exception that
    # supports "off".
    interval_steps: int = 500
    # TOTAL clip count per dataset per eval round, summed across all ranks
    # (training/eval_loaders.py shards this via DistributedSampler across every
    # rank, not just rank 0) - so this scales with world_size for a given
    # wall-clock budget, not a per-rank count.
    num_clips_per_dataset: int = 256
    datasets: list[str] = dataclasses.field(
        default_factory=lambda: ["how2sign", "phoenix2014t", "csl_daily"])


@dataclasses.dataclass
class Stage2Config:
    seed: int
    # One shared LR for the combined svit+heads+unet+tt optimizer (training/
    # stage2.py's run_pass_a/b/c each gate which subset actually moves via
    # their own requires_grad_ toggling) - matches Stage 1's single-LR
    # approach; the plan doesn't call for per-pass or per-component rates.
    learning_rate: float
    num_steps: int
    log_interval_steps: int
    checkpoint_interval_steps: int
    device: str
    # Stage 2's OWN checkpoint directory - separate from Stage 1's
    # checkpoint_dir. Stage 2 never writes into Stage 1's checkpoints.
    checkpoint_dir: str
    # One-time seed: svit+heads only (no optimizer, no unet/tt) loaded from a
    # Stage 1 checkpoint at startup, ONLY when this run has no Stage-2-own
    # checkpoint to resume from (--checkpoint_pth unset). Ignored on a resume,
    # since Stage 2's own checkpoint already has svit/heads as they were
    # mid-Stage-2-training, not Stage 1's original values.
    stage1_checkpoint_pth: str
    dataloader_config_path: str
    # See PretrainConfig.num_expression_params' docstring - must match whatever
    # stage1_checkpoint_pth's (and, on a resume, checkpoint_pth's) own dim was
    # trained with; training/checkpoint.py validates this on load rather than
    # silently mismatching. 100 (default) reproduces the original,
    # pre-configurable behavior exactly for every existing YAML.
    num_expression_params: int = 100
    # Sec 7: "Cycle through the three passes each iteration (or in a fixed
    # pattern; make the pattern a config knob)" - default is plain round-robin,
    # but exposed as a real list (not hardcoded) per that explicit instruction.
    pass_pattern: list[str] = dataclasses.field(default_factory=lambda: ["A", "B", "C"])
    # Pass B cycles through three modes across CONSECUTIVE calls to Pass B
    # specifically (not outer-loop steps overall): pass_b_encoder_steps calls
    # updating (tokens+SViT+heads) only, then pass_b_unet_steps calls updating
    # UNet only, then pass_b_joint_steps calls updating both together - then
    # repeats. Defaults (1, 1, 0) reproduce the original SMIRK-style pure
    # alternation (period 1, no joint mode). Setting e.g. (1, 0, 0) gives
    # encoder-only Pass B; (0, 0, 1) gives fully joint Pass B every call. See
    # run_pass_b's docstring for the mode semantics and stage2.py's train()
    # for the cycle-position -> mode mapping.
    pass_b_encoder_steps: int = 1
    pass_b_unet_steps: int = 1
    pass_b_joint_steps: int = 0
    # Linear LR ramp from learning_rate/warmup_steps up to the full
    # learning_rate over the first warmup_steps steps, then flat. Added after
    # observing landmark loss jump ~100x within 150 steps of Stage 2 starting -
    # Stage 2 reuses Stage 1's flat LR verbatim (no schedule/warmup either
    # stage) and builds a brand-new, freshly-reset Adam optimizer (no
    # momentum/variance carried over from Stage 1) right as several
    # large-magnitude, previously-nonexistent losses (photometric/VGG/emotion/
    # cycle, plus a randomly-initialized UNet) start contributing gradient to
    # an already-converged svit/heads - a flat full-LR Adam start is a shock
    # to that converged checkpoint. 0 disables warmup (flat LR throughout,
    # the old behavior). Driven entirely by the training loop's own `step`
    # counter (training/stage2.py's train()), which already resumes correctly
    # from a checkpoint - so this never re-triggers on a resume past
    # warmup_steps, no separate state needs saving/loading.
    warmup_steps: int = 1000
    # Keeps svit/heads frozen (requires_grad_(False)) in Pass A regardless of
    # that pass's own default-unfreeze behavior - see run_pass_a's own
    # docstring. Intended for a UNet-warmup phase: train the reconstruction
    # pathway (photometric/VGG/emotion, all routed through unet) against a
    # STABLE geometry signal - normally svit/heads/unet all move together in
    # Pass A, which means unet is chasing a moving target on top of everything
    # else destabilizing at once. Only meaningful when pass_pattern excludes
    # "B"/"C" (both would still move the encoder) - e.g. pass_pattern: [A].
    # False (default) preserves the original always-unfrozen-in-Pass-A
    # behavior.
    freeze_encoder: bool = False
    datasets_yaml_path: str = str(DEFAULT_DATASETS_YAML)
    # Resume Stage 2's own training from this checkpoint file - unlike
    # stage1_checkpoint_pth (one-time seed), this is checked on every run.
    # None (unset in the YAML) means a fresh Stage 2 run.
    checkpoint_pth: str | None = None
    # When checkpoint_pth is set, whether to also resume its optimizer state
    # and step count (True, original behavior) or just seed svit/heads/unet/tt
    # weights from it and start fresh at step 0 with a new optimizer (False) -
    # e.g. seeding from a unet-warmup-phase checkpoint whose optimizer
    # momentum only ever moved unet (svit/heads were frozen that whole phase,
    # see freeze_encoder), which would be stale/meaningless to carry into a
    # phase where they move again. Ignored when checkpoint_pth is None.
    resume_optimizer_and_step: bool = True
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    # Pass C synthetic occlusion (implementation-plan.md's "out of scope for
    # now" list, Sec 7): while enabled, some real, visible frames in each Pass
    # C clip are pretended-occluded for TT's input only (reuses model/
    # encoding.py's own _fill_missing_frame_tokens neighbor-averaging - the
    # SViT tokens for a chosen frame are replaced by the average of its
    # nearest good neighbors and its visibility score fed to TT is zeroed),
    # while still supervising against that frame's real, uncorrupted target -
    # teaches TT to recover a frame's parameters from temporal context alone.
    # False (default) disables the feature entirely; the two fields below are
    # ignored unless this is True.
    pass_c_synthetic_occlusion_enabled: bool = False
    # Per-eligible-frame probability of being chosen as synthetically
    # occluded, independently per frame (no multi-frame bursts). 0.1: with
    # TT_WINDOW_SIZE=15 (model/constants.py, radius 7 either side), a 10%
    # independent per-frame rate means ~77% of any given frame's local
    # attention window contains at least one occluded neighbor (1 - 0.9^14),
    # giving the augmentation regular exposure across windows, while ~90% of
    # frames overall stay clean - a starting point, not a calibrated value;
    # tune empirically once training is running.
    pass_c_synthetic_occlusion_prob: float = 0.1
    # Multiplier on synthetically-occluded frames' contribution to Pass C's
    # losses (2D reconstruction + temporal smoothness) - e.g. 2.0 means an
    # occluded frame contributes 2x a normal frame's loss, on top of its
    # normal 1x share. 1.0 is a no-op.
    pass_c_occlusion_loss_weight: float = 2.0
    # occlusion-experiment1.md's Change 1: drop (zero-weight) individual landmarks
    # whose GT 2D position falls in the occluded region of the frame (model/losses/
    # landmark.py's landmark_visibility_mask), plus the matching per-PAIR masking
    # for eye/lip closure terms - via _compute_2d_reconstruction_losses_from_encoded's
    # own landmark_occlusion_masking parameter, threaded into BOTH Pass A's and Pass
    # C's call sites (occlusion-experiment1-passA.md extended this from Pass-C-only
    # once Pass A started alternating back into the pass_pattern). False (default)
    # reproduces the exact original unmasked landmark/closure loss everywhere.
    landmark_occlusion_masking: bool = False
    # Change 2: peak weight for the gated expressive vertex regions (lips,
    # eye_region, left/right_eye_region, nose, forehead) under full occlusion
    # (gate=1) - model/losses/temporal_smoothness.py's vertex_velocity_penalty.
    # Inert whenever temporal_vertex_smoothness_weight is 0 (the default),
    # regardless of this field's own value - safe to default to the spec's own
    # starting value (3.0) without affecting any config that leaves
    # temporal_vertex_smoothness_weight at 0.
    expressive_region_smooth_weight: float = 3.0
    # Change 2: overall weight of the new vertex-space, region-weighted,
    # visibility-gated temporal smoothness term. 0.0 (default) fully disables the
    # term (no separate "enabled" boolean exists for it) - this default, unlike
    # expressive_region_smooth_weight's, MUST be 0.0 (not the spec's per-experiment
    # value of 0.1) so every existing/other Stage2 YAML that doesn't set this field
    # keeps training with NO vertex-space smoothness term at all, exactly as before
    # this change. This experiment's own YAML sets it to 0.1.
    temporal_vertex_smoothness_weight: float = 0.0
    # Change 2: gates the EXISTING param-space velocity term (expression/jaw/
    # camera/shape, training/stage2.py's compute_temporal_smoothness_losses) on/off
    # in Pass C. True (default) reproduces the original always-on behavior; this
    # experiment's own YAML sets it False (the vertex-space term is the ONLY
    # temporal smoothness mechanism active in Pass C for this run).
    param_smoothness_in_pass_c: bool = True
    # Change 2 optional refinement: use a mouth-region-specific visibility
    # (fraction of lip landmark points - model/losses/landmark.py's
    # mouth_point_indices - not occluded per landmark_visibility_mask) instead of
    # the whole-face visibility_ratio score for the gate that drives Change 2's
    # weighting. False (default, and this experiment's own starting value too, per
    # the spec's "keep whole-face score as default to limit moving parts in the
    # first run") uses the whole-face score.
    mouth_gate_use_region_visibility: bool = False
    # Pass C identity pooling: replaces encode_video's per-frame decoded
    # `shape` (FLAME identity) with a per-clip masked mean, broadcast back to
    # every real frame (model/encoding.py's _pool_identity) - so identity is
    # architecturally constant within a clip rather than only softly
    # discouraged from drifting via the existing param-space velocity term.
    # Unweighted by visibility, deliberately: TT's own attention is assumed to
    # already account for visibility when refining tokens, so a second
    # re-weighting here would be redundant. Also threaded into
    # run_periodic_eval_local's own encode_video call, so eval reflects the
    # same behavior training is optimizing. False (default) reproduces the
    # original per-frame shape behavior exactly - see stage2-config-
    # reference.md for the full rationale and data behind this experiment.
    pass_c_identity_pooling: bool = False
    # Which formula computes Change 2's vertex-space gate (model/losses/
    # temporal_smoothness.py's compute_vertex_gate) - "min_vis" (default,
    # gate = 1 - min(gate_signal[t], gate_signal[t+1])) reproduces the
    # original formula exactly, but is poorly calibrated against this
    # project's real visibility_ratio data (never near 1.0 even on clean
    # frames). "delta_vis" gates on the frame-to-frame CHANGE in gate_signal
    # instead (better calibrated - see vertex_gate_delta_cap/_beta below and
    # stage2-config-reference.md), targeting occlusion onset/offset
    # specifically rather than sustained mid-occlusion smoothing.
    vertex_gate_mode: str = "min_vis"
    # "delta_vis" mode only (ignored otherwise): normalizes the raw
    # frame-to-frame |gate_signal[t] - gate_signal[t+1]| diff against this cap
    # before clamping to [0, 1] - a diff at/above cap saturates to a gate of
    # 1.0. 0.1 is roughly the global p95 frame-to-frame diff measured across
    # csl_daily/how2sign/phoenix2014t's cached visibility_ratio - see
    # stage2-config-reference.md for the full percentile breakdown.
    vertex_gate_delta_cap: float = 0.1
    # "delta_vis" mode only (ignored otherwise): gate = norm ** beta, where
    # norm is the cap-normalized diff above. beta=1 (default) is a no-op
    # (gate=norm, plain linear). beta > 1 suppresses small/moderate norm
    # values much faster than large ones (norm already near 1 barely changes
    # under any power), widening the separation between genuine transitions
    # and ordinary jitter as beta increases - i.e. higher beta means
    # higher-diff frames dominate more over low-diff ones, not less. Named
    # `beta`, not `gamma`, specifically to avoid confusion with the
    # unrelated, oppositely-behaved tt_gamma below (which reshapes raw
    # per-key visibility inside GatedTemporalTransformer's own attention, not
    # this loss-side gate). See stage2-config-reference.md for the measured
    # gate-value distribution at several candidate beta values.
    vertex_gate_delta_beta: float = 1.0
    # Restricts Pass C's own clip_loader (never frame_pool_loader/eval loaders)
    # to only occlusion-positive clip windows - scripts/build_occlusion_index.py's
    # output directory, one <dataset_name>.jsonl per registered 2d_video dataset,
    # each line {"sample_id": ..., "start": ...} (dataset_processing/dataloading/
    # occlusion_index.py's is_window_occlusion_positive decides which windows
    # qualify, using the same |Δvisibility| definition as vertex_gate_mode=
    # "delta_vis" above). Motivation: most sampled clips in an ordinary batch
    # contain no occlusion event at all, so even the delta_vis-gated smoothness
    # loss rarely fires within a given batch - restricting Pass C's own training
    # population to occlusion-containing windows makes it fire on (close to)
    # every batch instead. A dataset with no matching <name>.jsonl in this
    # directory is left completely unfiltered (dataset_processing/dataloading/
    # combined_loader.py's _load_occlusion_index). None (default) disables this
    # entirely - every existing YAML keeps drawing Pass C's clips from the full,
    # unfiltered train split exactly as before.
    pass_c_occlusion_subset_index_dir: str | None = None
    # Per-run override for constants.TEMPORAL_VELOCITY_WEIGHT (the fixed module-
    # level constant weighting Pass C's param-space velocity term - vel_expr/
    # vel_jaw/vel_camera/vel_shape, gated on/off by param_smoothness_in_pass_c
    # above), so an experiment YAML can tune it without changing the shared
    # constant every other TEMPORAL_VELOCITY_WEIGHT reference would also pick
    # up. Defaults to the constant itself (0.1), reproducing the exact
    # original behavior for every existing YAML that doesn't set this field.
    temporal_velocity_weight_in_pass_c: float = constants.TEMPORAL_VELOCITY_WEIGHT
    # Which TT architecture to use (model/temporal.py) - "original" (default,
    # TemporalTransformer: 7 QK+ALiBi heads + 1 dedicated visibility-only head),
    # "simple" (SimpleTemporalTransformer: all 8 heads QK+ALiBi, no visibility
    # input at all), or "gated" (GatedTemporalTransformer: all 8 heads QK+ALiBi,
    # with a uniform post-softmax visibility gate + renormalize applied to every
    # head). Changing this away from "original" means checkpoint_pth's saved tt
    # weights won't shape-match the new architecture's q_proj/k_proj (sized
    # differently per variant) - train() excludes "tt" from that checkpoint's
    # load in that case, so svit/heads/unet still warm-start from checkpoint_pth
    # but tt itself starts fresh (identity-at-init, same as any new run).
    tt_variant: str = "original"
    # GatedTemporalTransformer's post-softmax visibility-gating exponent
    # (GatedTTConfig.gamma) - a real config knob rather than a hardcoded default,
    # so different experiment YAMLs can sweep it without a code change. Ignored
    # unless tt_variant == "gated".
    tt_gamma: float = constants.TT_GATE_GAMMA
    # wandb (training/wandb_utils.py) run identity - all optional so existing
    # YAMLs need no changes. wandb_run_name=None lets wandb auto-generate a name.
    wandb_project: str = "msc-project-stage2"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


def load_stage2_config(path: str | Path) -> Stage2Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    # Mirrors load_dataloader_config's DetectorConfig(**raw["detector"]) pattern:
    # a plain **raw unpack would leave `eval` as a bare dict, not an EvalConfig,
    # if the YAML supplies one.
    if "eval" in raw:
        raw["eval"] = EvalConfig(**raw["eval"])
    cfg = Stage2Config(**raw)
    if not 50 <= cfg.num_expression_params <= 100:
        raise ValueError(f"num_expression_params must be between 50 and 100, got {cfg.num_expression_params}")
    if "B" in cfg.pass_pattern:
        cycle_length = cfg.pass_b_encoder_steps + cfg.pass_b_unet_steps + cfg.pass_b_joint_steps
        if cycle_length <= 0:
            raise ValueError(
                "pass_b_encoder_steps + pass_b_unet_steps + pass_b_joint_steps must be > 0 when "
                "pass_pattern includes 'B'"
            )
    return cfg
