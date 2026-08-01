"""Regularization losses (implementation-plan.md Sec 6, "all passes"): L2 penalty
pulling shape/expression/jaw parameters back toward zero, preventing degenerate/
runaway values. FLAME's shape/expression bases are zero-centered PCA coefficients,
so zero already means "neutral"/"mean face" - there's no separate target to regress
toward, unlike e.g. SMIRK's optional (and non-default) "regularize toward a base
model" mode, which isn't implemented here since this project has no analogous
frozen-reference-model concept and SMIRK itself defaults to the zero-target version.

Pattern adapted from SMIRK (Retsinas et al., CVPR 2024,
https://github.com/georgeretsi/smirk, src/smirk_trainer.py, MIT License, Copyright
(c) 2024 George Retsinas) - inlined there as 3 near-identical lines in the trainer,
factored out here as one reusable function. Weights (model/constants.py) follow
TokenFace's single uniform value (1e-4) across all three parameter groups, not
SMIRK's own differentiated per-group weights - see constants.py for why.

log_scale_regularization is a separate, later addition targeting camera scale
specifically - added after observing scale drift well below its Stage-1-
converged value (~7.4) across multiple Stage 2 runs, with no loss term
anywhere previously anchoring it (unlike shape/expression/jaw, camera has
zero direct ground-truth supervision). Deliberately NOT l2_regularization
toward 0: scale is a strictly-positive, multiplicative quantity, so 0 is its
degenerate collapse point, not a neutral reference the way 0 is for
zero-centered PCA coefficients - regularizing toward 0 would reward exactly
the failure mode being prevented. log(scale/reference) fixes this: zero
penalty at scale==reference, symmetric penalty for proportionally-equal
over/under-shoot (unlike raw L2's quadratic asymmetry), and an unbounded
penalty as scale->0 (steepens exactly where protection is needed, unlike raw
L2 whose gradient shrinks toward 0 as scale->0).
"""

from __future__ import annotations

import torch


def l2_regularization(params: torch.Tensor) -> torch.Tensor:
    """params: (B, D) -> scalar mean squared value - an L2 penalty pulling params
    toward zero (not toward any other target)."""
    return torch.mean(params**2)


def log_scale_regularization(scale: torch.Tensor, reference: float = 7.0) -> torch.Tensor:
    """scale: (B, 1) or (B,) camera scale -> scalar L2 penalty on log(scale/reference),
    pulling scale toward `reference` (see module docstring for why not toward 0).
    reference=7.0 matches both SMIRK's own hand-chosen camera-scale init constant
    (src/smirk_encoder.py's PoseEncoder) and this project's own Stage 1 pretrain
    endpoint (~7.4) for the same weak-perspective camera convention.

    clamp(min=1e-6): pure numerical safety - scale has no positivity constraint
    anywhere upstream (a plain nn.Linear output), so a literal non-positive value
    is possible even if not yet observed; without the clamp, log() of that would
    be NaN, corrupting this term right in the collapse case it exists to guard
    against."""
    return torch.mean(torch.log(scale.clamp(min=1e-6) / reference) ** 2)
