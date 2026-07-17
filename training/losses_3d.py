"""3D-batch loss computation (mesh + Lvc, implementation-plan.md Sec 6/7),
identical between Stage 1 and Stage 2 Pass A per Sec 7's own spec ("3D batches:
mesh + Lvc" appears unchanged in both stages' loss lists) - shared here rather
than duplicated in both training/pretrain.py and training/stage2.py."""

from __future__ import annotations

import torch

from model import constants
from model.encoder import SViT
from model.encoding import encode_image
from model.flame.flame import FLAME
from model.heads import ComponentHeads
from model.losses.mesh import region_weighted_mesh_loss, vertex_consistency_loss
from training.loss_utils import regularization_loss


def find_identity_pairs(subject_ids: list[str]) -> list[tuple[int, int]]:
    """Groups indices by subject_id, forms non-overlapping consecutive pairs
    within each group (a group of 4 -> 2 pairs, not all C(4,2)=6 combinations -
    avoids redundant compute and any one sample appearing in multiple pairs).
    IdentityAwareBatchSampler already guarantees ~half the 3D batch is composed
    of such pairs; this just finds them (the sampler communicates nothing
    beyond subject_id itself, per its own docstring)."""
    groups: dict[str, list[int]] = {}
    for index, subject_id in enumerate(subject_ids):
        groups.setdefault(subject_id, []).append(index)

    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for i in range(0, len(indices) - 1, 2):
            pairs.append((indices[i], indices[i + 1]))
    return pairs


def compute_3d_losses(
    svit: SViT, heads: ComponentHeads, flame: FLAME, region_weights: torch.Tensor,
    batch_3d: dict[str, torch.Tensor], subject_ids: list[str], device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_image(svit, heads, batch_3d["pixel_values"])
    flame_out = flame(encoded["shape"], encoded["expression"], encoded["jaw"], encoded["eyelid"], encoded["rotation"])

    mesh_loss = region_weighted_mesh_loss(flame_out["vertices"], batch_3d["flame_vertices"], region_weights)

    pairs = find_identity_pairs(subject_ids)
    if pairs:
        idx_a = torch.tensor([p[0] for p in pairs], device=device)
        idx_b = torch.tensor([p[1] for p in pairs], device=device)

        # Direction 1: a's shape + b's expression/jaw/eyelid/rotation, compared
        # against b's own GT (TokenFace Eq. 5's literal direction).
        swapped_a_into_b = flame(
            encoded["shape"][idx_a], encoded["expression"][idx_b], encoded["jaw"][idx_b],
            encoded["eyelid"][idx_b], encoded["rotation"][idx_b],
        )["vertices"]
        # Direction 2: the reverse - b's shape + a's motion, compared against a's
        # own GT. Not in the plan's own Eq. 5, but the pair is already found, so
        # this doubles the Lvc signal per pair at negligible extra cost.
        swapped_b_into_a = flame(
            encoded["shape"][idx_b], encoded["expression"][idx_a], encoded["jaw"][idx_a],
            encoded["eyelid"][idx_a], encoded["rotation"][idx_a],
        )["vertices"]

        lvc_loss = (
            vertex_consistency_loss(swapped_a_into_b, batch_3d["flame_vertices"][idx_b], region_weights)
            + vertex_consistency_loss(swapped_b_into_a, batch_3d["flame_vertices"][idx_a], region_weights)
        ) / 2
    else:
        lvc_loss = torch.zeros((), device=device)

    reg_loss = regularization_loss(encoded)

    total = constants.MESH_LOSS_LAMBDA * mesh_loss + constants.VERTEX_CONSISTENCY_LOSS_LAMBDA * lvc_loss + reg_loss
    metrics = {
        "mesh": mesh_loss.item(),
        "lvc": lvc_loss.item() if torch.is_tensor(lvc_loss) else lvc_loss,
        "reg_3d": reg_loss.item(),
        "num_pairs": len(pairs),
    }
    return total, metrics
