"""Expression + identity cycle consistency (implementation-plan.md Sec 6:
"Expression cycle consistency", "Identity (β) cycle consistency"; Sec 7's
augmentation pass).

Adapted from SMIRK (Retsinas et al., CVPR 2024, https://github.com/georgeretsi/smirk,
src/smirk_trainer.py's step2(), src/utils/utils.py's load_templates(), and
src/base_trainer.py's load_random_template(); MIT License, Copyright (c) 2024
George Retsinas) - cross-checked against SMIRK's own paper
(arxiv.org/abs/2404.04104) for the exact loss formula.

Eq. 2 (the paper's named "Expression Consistency" loss) covers expression only:
Lexp = ||Eψ(T(R(θ,β,ψaug) ⊕ M(I))) − ψaug||². Jaw/eyelid consistency terms are
SMIRK's own code-level addition beyond that formula (the paper's prose only says
augmentations "simultaneously simulate jaw and eyelid openings/closings" for
realism, without a separate named loss for them) - expression_cycle_loss below
follows the actual code (expression + jaw*10 + eyelid*10), which is what this
plan's "with jaw+eyelid co-augmentation" phrasing and single outer "cycle 10"
weight are calibrated against, not the paper's narrower written equation.

The paper also confirms this plan's noted deviation for identity/shape consistency:
"since the shape encoder Eβ is frozen [in SMIRK], the consistency loss only affects
the optimization of the translator". This project's shape pathway isn't frozen, so
identity_cycle_loss is meant to be applied on both alternations (encoder update and
UNet update) - an additional encoder disentanglement signal SMIRK's own design
couldn't use.

Four augmentation types (SMIRK's own "Promoting Diverse Expressions" recipe),
randomly assigned per-sample by splitting a batch into 4 equal groups: perturbation
(jitter ~half the expression dims with substantial noise), permutation (within the
group, borrow another group member's expression, intensity-scaled), template
injection (inject a real, precomputed FaMoS-fitted expression), and zero-expression
- plus jaw and eyelid co-augmentation applied (mildly) to every sample regardless
of group, with the zero-expression group's jaw/eyelid then further, more
aggressively overridden (jaw forced to exactly neutral, eyelid to fully random) -
per the paper: "more aggressive augmentations in the zero-expression case to avoid
incompatible blending with intense expressions". The augmentation formulas' numeric
constants (noise scales, clamp ranges, etc.) are SMIRK's own tuned recipe,
reproduced as-is and kept inline (not hoisted to model/constants.py) since they're
tightly coupled to their specific formula lines, not independently meaningful
config knobs.

Template injection needs SMIRK's own precomputed FaMoS-fitted expression templates
(assets/expression_templates_famos/, Sec 5.3) - load_expression_templates() loads
them from disk once; sample_random_template() and the augmentation function itself
take the loaded templates dict as a parameter rather than re-loading it internally.
These templates only have 50 expression dims (SMIRK's own encoder config only
ever used num_expression=50, so that's all their fitting pipeline saved), not our
full FLAME_EXPRESSION_DIM=100 - our own indexed FaMoS data (dataset_processing/
manifests/famos.jsonl) is larger but only has raw registered meshes, not FLAME
parameters, so fitting our own higher-dim templates would mean building a
mesh-to-FLAME optimization pipeline from scratch, out of scope here. Resolved by
zeroing dims 50-99 for this augmentation type specifically (_augment_template_
injection), rather than leaving them at their pre-augmentation value or fabricating
nonzero data we have no basis for.

Both augment_expression_cycle()'s outputs are detached: they're meant to be fixed
targets for the cycle loss (like labels), not a differentiable path back into
whatever produced the original expression/jaw/eyelid (the first-pass encoder
output) - several of the augmentation formulas are linear combinations that
include the original tensor (e.g. perturbation adds noise to `expression` itself),
so without detaching, cycle_loss's gradient would leak into the first-pass encoder
through the *target* side. Only the re-encoded second-pass prediction
(recon_expression/recon_jaw/recon_eyelid, computed by the caller) should receive
gradient from this loss.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import constants

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_expression_templates(
    templates_path: str | Path = _REPO_ROOT / constants.EXPRESSION_TEMPLATES_PATH,
) -> dict[str, np.ndarray]:
    """Loads SMIRK's precomputed FaMoS-fitted expression templates. Returns
    dict[subject+template_class] -> (num_frames, n_exp) array of expression
    coefficients. Matches SMIRK's utils.load_templates() exactly."""
    templates_path = Path(templates_path)
    templates: dict[str, np.ndarray] = {}
    for subject in os.listdir(templates_path):
        subject_path = templates_path / subject
        if not subject_path.is_dir():
            continue
        for template_class in os.listdir(subject_path):
            if template_class.endswith(".mp4"):
                continue
            if template_class not in constants.EXPRESSION_TEMPLATE_CLASSES:
                continue
            class_path = subject_path / template_class
            exps = [
                np.load(class_path / npy_file, allow_pickle=True).item()["expression"].squeeze()
                for npy_file in os.listdir(class_path)
            ]
            templates[subject + template_class] = np.array(exps)
    return templates


def sample_random_template(templates: dict[str, np.ndarray], num_expression_params: int) -> np.ndarray:
    """Picks a random template class, then a random frame within it. Returns
    (num_expression_params,) expression coefficients, matching SMIRK's
    base_trainer.load_random_template()."""
    random_key = random.choice(list(templates.keys()))
    class_templates = templates[random_key]
    random_index = random.randint(0, class_templates.shape[0] - 1)
    return class_templates[random_index][:num_expression_params]


def _augment_perturbation(expression: torch.Tensor) -> torch.Tensor:
    """expression: (n_group, n_exp) - this group's sub-batch. Adds large random
    noise (scaled 1x-3x) to ~half of the expression dims (randomly chosen per
    sample), plus a small extra jitter, clamped to [-4, 4]."""
    n, feats_dim = expression.shape
    device = expression.device
    param_mask = torch.bernoulli(torch.ones((n, feats_dim), device=device) * 0.5)
    new_expression = torch.randn((n, feats_dim), device=device) * (1 + 2 * torch.rand((n, 1), device=device)) * param_mask + expression
    return torch.clamp(new_expression, -4.0, 4.0) + (0.2 * torch.rand((n, 1), device=device)) * torch.randn(
        (n, feats_dim), device=device
    )


def _augment_permutation(expression: torch.Tensor) -> torch.Tensor:
    """expression: (n_group, n_exp) - this group's sub-batch. Replaces each
    sample's expression with another sample's expression FROM WITHIN THIS SAME
    SUB-BATCH (a random permutation of just these n_group rows, not the full
    original batch), scaled by a random factor in [0.25, 1.5], plus a small extra
    jitter."""
    n, feats_dim = expression.shape
    device = expression.device
    permuted = expression[torch.randperm(n, device=device)]
    return (0.25 + 1.25 * torch.rand((n, 1), device=device)) * permuted + (
        0.2 * torch.rand((n, 1), device=device)
    ) * torch.randn((n, feats_dim), device=device)


def _augment_template_injection(
    expression: torch.Tensor, templates: dict[str, np.ndarray], num_expression_params: int
) -> torch.Tensor:
    """expression: (n_group, n_exp) - this group's sub-batch. Replaces each
    sample's expression with a real, precomputed FaMoS-fitted expression template,
    scaled by a random factor in [0.25, 1.5], plus a small extra jitter. SMIRK's
    own templates only cover the first 50 of our 100 expression dims (SMIRK's own
    encoder config only ever used num_expression=50) - dims beyond
    num_expression_params are zeroed rather than left at the original
    pre-augmentation value, so this augmented sample is a clean "template + jitter"
    target, not a template/original hybrid."""
    n, feats_dim = expression.shape
    device = expression.device
    new_expression = torch.zeros_like(expression)
    for i in range(n):
        template = torch.tensor(
            sample_random_template(templates, num_expression_params), dtype=expression.dtype, device=device
        )
        new_expression[i, :num_expression_params] = (0.25 + 1.25 * torch.rand((1,), device=device)) * template
    return new_expression + (0.2 * torch.rand((n, 1), device=device)) * torch.randn((n, feats_dim), device=device)


def _augment_zero_expression(expression: torch.Tensor) -> torch.Tensor:
    """expression: (n_group, n_exp) - this group's sub-batch. Zeroes the
    expression, then adds a small amount of noise around zero."""
    n, feats_dim = expression.shape
    device = expression.device
    return (0.2 * torch.rand((n, 1), device=device)) * torch.randn((n, feats_dim), device=device)


def _augment_jaw(jaw: torch.Tensor) -> torch.Tensor:
    """jaw: (N, 3) - the FULL batch (all groups), unlike the expression
    augmentations above. With 50% probability per sample, adds noise to jaw
    (full-scale on the open axis, 10% scale on the other two), then clamps the
    open axis to a physically valid [0, 0.5] range."""
    n = jaw.shape[0]
    device = jaw.device
    scale_mask = torch.tensor([1.0, 0.1, 0.1], device=device).view(1, 3) * torch.bernoulli(
        torch.ones(n, device=device) * 0.5
    ).view(-1, 1)
    jaw = jaw + torch.randn(jaw.shape, device=device) * 0.2 * scale_mask
    jaw = jaw.clone()
    jaw[..., 0] = torch.clamp(jaw[..., 0], 0.0, 0.5)
    return jaw


def _augment_eyelids(eyelid: torch.Tensor) -> torch.Tensor:
    """eyelid: (N, 2) - the FULL batch (all groups). Adds uniform noise in
    [-0.25, 0.25], clamped to the valid [0, 1] eyelid range."""
    device = eyelid.device
    eyelid = eyelid + (-1 + 2 * torch.rand(eyelid.shape, device=device)) * 0.25
    return torch.clamp(eyelid, 0.0, 1.0)


def augment_expression_cycle(
    expression: torch.Tensor,
    jaw: torch.Tensor,
    eyelid: torch.Tensor,
    templates: dict[str, np.ndarray],
    num_expression_params: int = constants.EXPRESSION_TEMPLATE_NUM_DIMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """expression: (N, n_exp), jaw: (N, 3), eyelid: (N, 2) - N is the batch size
    for the augmentation pass (the caller decides whether/how to tile it, e.g.
    SMIRK's own Ke-repeat scheme for multiple augmented versions per real sample;
    this function just operates on whatever N rows it's given).

    Splits the batch into 4 random equal groups, applying a different augmentation
    type to each group's sub-batch (perturbation, permutation, template injection,
    zero-expression - each function only sees/mixes within its own group). Then
    applies mild jaw/eyelid co-augmentation across the FULL batch, and finally
    overrides that mild result for the zero-expression group specifically with a
    more aggressive one (jaw forced to exactly neutral, eyelid set to fully
    random) - see module docstring for why.

    Returns (aug_expression, aug_jaw, aug_eyelid), all detached (see module
    docstring)."""
    n = expression.shape[0]
    device = expression.device
    group_ids = torch.randperm(n, device=device)
    quarter = n // 4
    groups = [
        group_ids[:quarter],
        group_ids[quarter : 2 * quarter],
        group_ids[2 * quarter : 3 * quarter],
        group_ids[3 * quarter :],
    ]

    aug_expression = expression.clone()
    aug_expression[groups[0]] = _augment_perturbation(expression[groups[0]])
    aug_expression[groups[1]] = _augment_permutation(expression[groups[1]])
    aug_expression[groups[2]] = _augment_template_injection(expression[groups[2]], templates, num_expression_params)
    aug_expression[groups[3]] = _augment_zero_expression(expression[groups[3]])

    aug_jaw = _augment_jaw(jaw)
    aug_eyelid = _augment_eyelids(eyelid)

    aug_jaw[groups[3]] = 0.0
    aug_eyelid[groups[3]] = torch.rand((len(groups[3]), eyelid.shape[1]), device=device)

    return aug_expression.detach(), aug_jaw.detach(), aug_eyelid.detach()


def expression_cycle_loss(
    recon_expression: torch.Tensor,
    aug_expression: torch.Tensor,
    recon_jaw: torch.Tensor,
    aug_jaw: torch.Tensor,
    recon_eyelid: torch.Tensor,
    aug_eyelid: torch.Tensor,
) -> torch.Tensor:
    """MSE(expression)*1 + MSE(jaw)*10 + MSE(eyelid)*10, matching SMIRK's actual
    code (see module docstring for why this differs from the paper's narrower
    Eq. 2). All *_params here should be the augmented targets (aug_*) and the
    re-encoded predictions from the cycle's second pass (recon_*)."""
    loss = constants.CYCLE_EXPRESSION_WEIGHT * F.mse_loss(recon_expression, aug_expression)
    loss = loss + constants.CYCLE_JAW_WEIGHT * F.mse_loss(recon_jaw, aug_jaw)
    loss = loss + constants.CYCLE_EYELID_WEIGHT * F.mse_loss(recon_eyelid, aug_eyelid)
    return loss


def identity_cycle_loss(recon_shape: torch.Tensor, original_shape: torch.Tensor) -> torch.Tensor:
    """MSE between the re-encoded shape (after the cycle) and the ORIGINAL
    (pre-augmentation) shape - shape is never itself augmented, only expression/
    jaw/eyelid are. Meant to be applied on both alternations of the augmentation
    pass (encoder update and UNet update) - see module docstring for why this
    differs from SMIRK's own (shape-encoder-frozen) usage."""
    return F.mse_loss(recon_shape, original_shape)
