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
import dataclasses
from pathlib import Path

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

from dataset_processing.dataloading.combined_loader import build_combined_loader
from dataset_processing.dataloading.config import load_dataloader_config
from model import constants
from model.encoder import SViT
from model.encoding import encode_image
from model.farl_weights import load_farl_pretrained
from model.flame.flame import FLAME
from model.flame.renderer import project_landmarks
from model.heads import ComponentHeads
from model.losses.landmark import eye_closure_loss, fan_boundary_loss, lip_closure_loss, mediapipe_landmark_loss
from model.losses.mesh import build_region_weights
from model.losses.mica_shape import mica_shape_loss
from training.checkpoint import load_checkpoint, save_checkpoint
from training.distributed import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from training.config import PretrainConfig, load_pretrain_config
from training.loss_utils import concat_category_fields, gated_loss, next_batch, regularization_loss, weighted_metrics
from training.losses_3d import compute_3d_losses
from training.wandb_utils import flatten_metrics

_REPO_ROOT_RELATIVE_FARL_PATH = "pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"


def compute_2d_losses(
    svit: SViT, heads: ComponentHeads, flame: FLAME, batch_2d: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_image(svit, heads, batch_2d["pixel_values"])
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
    reg_loss = regularization_loss(encoded)

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


def train(cfg: PretrainConfig, checkpoint_pth: str | None = None) -> None:
    torch.manual_seed(cfg.seed)
    rank, world_size, local_rank, device = setup_distributed(fallback_device=cfg.device)
    if is_main_process(rank):
        print(f"config: {cfg}")
        wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.wandb_run_name,
            job_type="stage1", config=dataclasses.asdict(cfg),
        )

    try:
        svit = SViT().to(device)
        load_farl_pretrained(svit, _REPO_ROOT_RELATIVE_FARL_PATH)
        heads = ComponentHeads(expression_dim=cfg.num_expression_params).to(device)
        flame = FLAME(n_exp=cfg.num_expression_params).to(device)
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
        start_step = 0
        if checkpoint_pth is not None:
            loaded_step = load_checkpoint(
                checkpoint_pth, {"svit": svit, "heads": heads}, optimizer, device,
                expected_num_expression_params=cfg.num_expression_params,
            )
            start_step = loaded_step + 1

        if is_distributed():
            # device_ids=[0], not [local_rank]: setup_distributed now restricts
            # CUDA_VISIBLE_DEVICES to exactly this rank's own GPU, so every
            # process's own device is always index 0 in its own restricted view,
            # regardless of local_rank's original torchrun-assigned value.
            svit = DistributedDataParallel(svit, device_ids=[0])
            heads = DistributedDataParallel(heads, device_ids=[0])

        dataloader_cfg = load_dataloader_config(cfg.dataloader_config_path)
        loader = build_combined_loader(
            dataloader_cfg, split="train", rank=rank, world_size=world_size,
            datasets_yaml_path=cfg.datasets_yaml_path, video_mode="frame_pool",
        )

        # On a resume, start the loader's own epoch counter from an
        # approximation (step // len(loader)) rather than 0 - avoids reusing
        # the exact same early shuffle orders after a resume (training/
        # checkpoint.py's save_checkpoint docstring has the full reasoning).
        epoch = start_step // len(loader)
        loader.set_epoch(epoch)
        iterator = iter(loader)

        keys_2d = ["pixel_values", "mica_shape", "flag_mica_valid", "landmarks_fan", "flag_landmarks_fan_valid", "landmarks_mp", "flag_landmarks_mp_valid"]
        keys_3d = ["pixel_values", "flame_vertices"]

        # Pre-set in case cfg.num_steps <= start_step (a resume where the
        # loop body never runs even once) - a for loop that never executes
        # never binds its loop variable, so referencing `step` after the loop
        # for the unconditional final-save below would otherwise raise
        # NameError.
        step = start_step - 1

        for step in range(start_step, cfg.num_steps):
            batch, iterator, epoch = next_batch(loader, iterator, epoch)
            batch_2d = concat_category_fields(batch, ["2d_image", "2d_video"], keys_2d, device)
            batch_3d = concat_category_fields(batch, ["3d_image"], keys_3d, device)
            subject_ids_3d = batch["3d_image"]["subject_id"]

            loss_2d, metrics_2d = compute_2d_losses(svit, heads, flame, batch_2d)
            loss_3d, metrics_3d = compute_3d_losses(svit, heads, flame, region_weights, batch_3d, subject_ids_3d, device)
            total_loss = constants.LOSS_BALANCE_2D * loss_2d + constants.LOSS_BALANCE_3D * loss_3d

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if is_main_process(rank) and step % cfg.log_interval_steps == 0:
                print(
                    f"step {step} (epoch {epoch}): total={total_loss.item():.4f} "
                    f"2d={metrics_2d} 3d={metrics_3d} "
                    f"2d_weighted={weighted_metrics(metrics_2d)} 3d_weighted={weighted_metrics(metrics_3d)}"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "total": total_loss.item(),
                            **flatten_metrics(metrics_2d, "2d"),
                            **flatten_metrics(metrics_3d, "3d"),
                        },
                        step=step,
                    )

            if is_main_process(rank) and (step + 1) % cfg.checkpoint_interval_steps == 0:
                path = save_checkpoint(
                    cfg.checkpoint_dir, step, {"svit": svit, "heads": heads}, optimizer,
                    num_expression_params=cfg.num_expression_params,
                )
                print(f"saved checkpoint: {path}")

        # Unconditional final save: cfg.num_steps isn't guaranteed to be a
        # multiple of checkpoint_interval_steps (and either could change
        # later), so the interval check above alone could finish training
        # without ever saving the final weights. A duplicate save (if the
        # last loop iteration's interval check ALSO just fired for this same
        # step) just overwrites the same file with identical content -
        # harmless. Guarded by step >= start_step so the num_steps<=start_step
        # edge case (loop never ran) skips this redundant save entirely.
        if is_main_process(rank) and step >= start_step:
            path = save_checkpoint(
                cfg.checkpoint_dir, step, {"svit": svit, "heads": heads}, optimizer,
                num_expression_params=cfg.num_expression_params,
            )
            print(f"saved final checkpoint: {path}")

        if is_main_process(rank):
            print("DONE!")
    finally:
        if is_main_process(rank) and wandb.run is not None:
            wandb.finish()
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 pretraining (implementation-plan.md Sec 7).")
    parser.add_argument("--config", type=str, default="training/config/pretrain.yaml")
    args = parser.parse_args()

    cfg = load_pretrain_config(args.config)
    if cfg.wandb_run_name is None:
        cfg.wandb_run_name = Path(cfg.checkpoint_dir).name
    train(cfg, checkpoint_pth=cfg.checkpoint_pth)


if __name__ == "__main__":
    main()
