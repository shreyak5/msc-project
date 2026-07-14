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
"""

from __future__ import annotations

import torch


def l2_regularization(params: torch.Tensor) -> torch.Tensor:
    """params: (B, D) -> scalar mean squared value - an L2 penalty pulling params
    toward zero (not toward any other target)."""
    return torch.mean(params**2)
