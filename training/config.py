"""Stage 1/Stage 2 training config (implementation-plan.md Sec 7), mirroring
dataset_processing/dataloading/config.py's dataclass + YAML-loader pattern
(which itself holds multiple related config dataclasses in one file)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML


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
    # Resume training from this checkpoint file - checked on every run. None
    # (unset in the YAML) means a fresh run.
    checkpoint_pth: str | None = None


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return PretrainConfig(**raw)


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
    # Sec 7: "Cycle through the three passes each iteration (or in a fixed
    # pattern; make the pattern a config knob)" - default is plain round-robin,
    # but exposed as a real list (not hardcoded) per that explicit instruction.
    pass_pattern: list[str] = dataclasses.field(default_factory=lambda: ["A", "B", "C"])
    # Pass B alternates (tokens+SViT+heads)-update vs UNet-update - this is
    # how many CONSECUTIVE calls to Pass B specifically (not outer-loop steps
    # overall) happen before flipping which side updates. Default 1 =
    # alternates every single Pass B call.
    pass_b_alternation_period: int = 1
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


def load_stage2_config(path: str | Path) -> Stage2Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    # Mirrors load_dataloader_config's DetectorConfig(**raw["detector"]) pattern:
    # a plain **raw unpack would leave `eval` as a bare dict, not an EvalConfig,
    # if the YAML supplies one.
    if "eval" in raw:
        raw["eval"] = EvalConfig(**raw["eval"])
    return Stage2Config(**raw)
