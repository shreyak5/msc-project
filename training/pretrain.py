"""Stage 1 pretraining loop (implementation-plan.md Sec 7, "Stage 1 -
Pre-training"): stabilizes SViT (incl. position embeddings), the component
tokens, and the MLP heads before the UNet/TT ever train against them.

Single-GPU (plain `python training/pretrain.py`), single-node multi-GPU, and
multi-node multi-GPU (`torchrun --nnodes=... --nproc_per_node=... ...`) all
work unchanged - see training/distributed.py's setup_distributed for how
that's detected, and slurm_jobs/pretrain.sh for the multi-node launch/
rendezvous setup.

Not built here at all (unlike Stage 2): the Renderer/rasterizer (no
photometric loss in Stage 1), the UNet, the TT, a live MICA model, or live
FAN/MediaPipe predictors - Stage 1's losses (landmark + MICA for 2D batches,
mesh + Lvc for 3D batches, per Sec 7) only need SViT, ComponentHeads, and
FLAME; MICA/landmark targets are already precomputed and cached (Sec 5.3).
"""

from __future__ import annotations

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel

from dataset_processing.dataloading.combined_loader import build_combined_loader
from dataset_processing.dataloading.config import load_dataloader_config
from model import constants
from model.encoder import SViT
from model.farl_weights import load_farl_pretrained
from model.flame.flame import FLAME
from model.flame.renderer import project_landmarks
from model.heads import ComponentHeads
from model.losses.landmark import eye_closure_loss, fan_boundary_loss, lip_closure_loss, mediapipe_landmark_loss
from model.losses.mesh import build_region_weights, region_weighted_mesh_loss, vertex_consistency_loss
from model.losses.mica_shape import mica_shape_loss
from model.losses.regularization import l2_regularization
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from training.config import PretrainConfig, load_pretrain_config
from training.loss_utils import gated_loss

_REPO_ROOT_RELATIVE_FARL_PATH = "pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"


def _concat_category_fields(batch: dict, categories: list[str], keys: list[str], device: str) -> dict[str, torch.Tensor]:
    """batch: one yielded step from CombinedFaceLoader (dict keyed by category
    name). categories: which of that step's categories to combine (e.g.
    ["2d_image", "2d_video"]) - safe to concatenate along the batch dimension
    even though their configured batch_sizes differ (dataset_processing/config/
    dataloader.yaml), and safe for 3D categories specifically because
    IdentityAwareBatchSampler's identity pairs are already guaranteed within
    each category's own batch before this concatenation ever happens - combining
    afterward doesn't lose that guarantee, it just makes one bigger batch out of
    two already-valid ones. Returns a flat dict (not nested by category) - each
    key maps to one tensor spanning all combined samples, with the first
    category's rows first, second category's rows after (torch.cat preserves
    list order)."""
    return {key: torch.cat([batch[category][key] for category in categories], dim=0).to(device) for key in keys}


def _split_expression(expression_and_eyelid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        expression_and_eyelid[:, : constants.FLAME_EXPRESSION_DIM],
        expression_and_eyelid[:, constants.FLAME_EXPRESSION_DIM :],
    )


def _split_camera(camera: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return camera[:, constants.CAMERA_SCALE_SLICE], camera[:, constants.CAMERA_ROTATION_SLICE], camera[:, constants.CAMERA_TRANSLATION_SLICE]


def encode(svit: SViT, heads: ComponentHeads, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
    """pixel_values: (B, 3, H, W) -> dict of decoded FLAME/camera parameters,
    split into the pieces FLAME.forward()/project_landmarks actually take
    (expression's trailing eyelid dims and camera's scale/rotation/translation
    slices, per model/constants.py's fixed layout)."""
    features = svit(pixel_values)
    params = heads(features)
    expression, eyelid = _split_expression(params["expression"])
    scale, rotation, translation = _split_camera(params["camera"])
    return {
        "shape": params["shape"],
        "expression": expression,
        "eyelid": eyelid,
        "jaw": params["jaw"],
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def _regularization_loss(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    """Shape/expression/jaw only (Sec 6: "L2 on expression parameters (and
    standard FLAME param regularizers - shape, jaw)") - deliberately not camera:
    zero isn't a sensible prior for scale (degenerate) or rotation (would bias
    against genuinely non-frontal poses, which matter here given sign language
    video's real head-orientation variation), unlike shape/expression's
    zero-centered PCA coefficients where zero legitimately means "neutral"."""
    return (
        constants.REG_SHAPE_WEIGHT * l2_regularization(encoded["shape"])
        + constants.REG_EXPRESSION_WEIGHT * l2_regularization(encoded["expression"])
        + constants.REG_JAW_WEIGHT * l2_regularization(encoded["jaw"])
    )


def compute_2d_losses(
    svit: SViT, heads: ComponentHeads, flame: FLAME, batch_2d: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode(svit, heads, batch_2d["pixel_values"])
    cam_for_proj = torch.cat([encoded["scale"], encoded["translation"]], dim=-1)

    flame_out = flame(encoded["shape"], encoded["expression"], encoded["jaw"], encoded["eyelid"], encoded["rotation"])
    projected_fan = project_landmarks(flame_out["landmarks_fan"], cam_for_proj)
    projected_mp = project_landmarks(flame_out["landmarks_mp"], cam_for_proj)

    landmark_loss = gated_loss(
        fan_boundary_loss, batch_2d["flag_landmarks_fan_valid"], projected_fan, batch_2d["landmarks_fan"]
    ) + gated_loss(
        mediapipe_landmark_loss, batch_2d["flag_landmarks_mp_valid"], projected_mp, batch_2d["landmarks_mp"]
    )
    closure_loss = gated_loss(
        eye_closure_loss, batch_2d["flag_landmarks_mp_valid"], projected_mp, batch_2d["landmarks_mp"]
    ) + gated_loss(
        lip_closure_loss, batch_2d["flag_landmarks_mp_valid"], projected_mp, batch_2d["landmarks_mp"]
    )
    mica_loss = gated_loss(mica_shape_loss, batch_2d["flag_mica_valid"], encoded["shape"], batch_2d["mica_shape"])
    reg_loss = _regularization_loss(encoded)

    total = (
        constants.LANDMARK_LOSS_WEIGHT * landmark_loss
        + constants.CLOSURE_LOSS_WEIGHT * closure_loss
        + constants.MICA_SHAPE_LOSS_WEIGHT * mica_loss
        + reg_loss
    )
    metrics = {
        "landmark": landmark_loss.item(), "closure": closure_loss.item(),
        "mica": mica_loss.item(), "reg_2d": reg_loss.item(),
    }
    return total, metrics


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
    encoded = encode(svit, heads, batch_3d["pixel_values"])
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

    reg_loss = _regularization_loss(encoded)

    total = constants.MESH_LOSS_LAMBDA * mesh_loss + constants.VERTEX_CONSISTENCY_LOSS_LAMBDA * lvc_loss + reg_loss
    metrics = {
        "mesh": mesh_loss.item(),
        "lvc": lvc_loss.item() if torch.is_tensor(lvc_loss) else lvc_loss,
        "reg_3d": reg_loss.item(),
        "num_pairs": len(pairs),
    }
    return total, metrics


def train(cfg: PretrainConfig, checkpoint_pth: str | None = None) -> None:
    torch.manual_seed(cfg.seed)
    rank, world_size, local_rank, device = setup_distributed(fallback_device=cfg.device)

    try:
        svit = SViT().to(device)
        load_farl_pretrained(svit, _REPO_ROOT_RELATIVE_FARL_PATH)
        heads = ComponentHeads().to(device)
        flame = FLAME().to(device)
        region_weights = build_region_weights().to(device)

        # Built on the plain (not-yet-DDP-wrapped) parameters - this stays valid
        # after wrapping below, since DistributedDataParallel doesn't clone
        # parameter objects, it just wraps the module and adds hooks around the
        # same nn.Parameter objects the optimizer already references.
        optimizer = torch.optim.Adam(list(svit.parameters()) + list(heads.parameters()), lr=cfg.learning_rate)

        # Loaded before wrapping in DDP: DistributedDataParallel's own
        # constructor broadcasts rank 0's parameters to every other rank, so
        # loading a checkpoint into the plain model first guarantees every rank
        # starts from identical weights via that broadcast, on top of every
        # rank reading the same checkpoint file from shared storage.
        start_epoch, step = 0, 0
        if checkpoint_pth is not None:
            loaded_epoch, step = load_checkpoint(checkpoint_pth, svit, heads, optimizer, device)
            start_epoch = loaded_epoch + 1

        if is_distributed():
            svit = DistributedDataParallel(svit, device_ids=[local_rank])
            heads = DistributedDataParallel(heads, device_ids=[local_rank])

        dataloader_cfg = load_dataloader_config(cfg.dataloader_config_path)
        loader = build_combined_loader(
            dataloader_cfg, split="train", rank=rank, world_size=world_size,
            datasets_yaml_path=cfg.datasets_yaml_path, video_mode="frame_pool",
        )

        keys_2d = ["pixel_values", "mica_shape", "flag_mica_valid", "landmarks_fan", "flag_landmarks_fan_valid", "landmarks_mp", "flag_landmarks_mp_valid"]
        keys_3d = ["pixel_values", "flame_vertices"]

        for epoch in range(start_epoch, cfg.num_epochs):
            loader.set_epoch(epoch)
            for batch in loader:
                batch_2d = _concat_category_fields(batch, ["2d_image", "2d_video"], keys_2d, device)
                batch_3d = _concat_category_fields(batch, ["3d_image", "3d_video"], keys_3d, device)
                subject_ids_3d = batch["3d_image"]["subject_id"] + batch["3d_video"]["subject_id"]

                loss_2d, metrics_2d = compute_2d_losses(svit, heads, flame, batch_2d)
                loss_3d, metrics_3d = compute_3d_losses(svit, heads, flame, region_weights, batch_3d, subject_ids_3d, device)
                total_loss = constants.LOSS_BALANCE_2D * loss_2d + constants.LOSS_BALANCE_3D * loss_3d

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if is_main_process(rank) and step % cfg.log_interval_steps == 0:
                    print(
                        f"epoch {epoch} step {step}: total={total_loss.item():.4f} "
                        f"2d={metrics_2d} 3d={metrics_3d}"
                    )
                step += 1

            if is_main_process(rank) and (epoch + 1) % cfg.checkpoint_interval_epochs == 0:
                path = save_checkpoint(cfg.checkpoint_dir, epoch, step, svit, heads, optimizer)
                print(f"saved checkpoint: {path}")

        if is_main_process(rank):
            print("DONE!")
    finally:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 pretraining (implementation-plan.md Sec 7).")
    parser.add_argument("--config", type=str, default="training/config/pretrain.yaml")
    parser.add_argument("--checkpoint_pth", type=str, default=None, help="Resume training from this checkpoint file")
    args = parser.parse_args()

    cfg = load_pretrain_config(args.config)
    train(cfg, checkpoint_pth=args.checkpoint_pth)


if __name__ == "__main__":
    main()
