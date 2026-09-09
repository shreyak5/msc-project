from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dataset_processing.dataloading.combined_loader import build_combined_loader
from dataset_processing.dataloading.config import load_dataloader_config
from evaluation.metrics import per_frame_euclidean_error, per_frame_vertex_error, summarize
from model import constants
from model.config import GatedTTConfig
from model.emotion.emotion_net import EmotionNet
from model.encoder import SViT
from model.encoding import encode_image, encode_video
from model.farl_weights import load_farl_pretrained
from model.flame.flame import FLAME
from model.flame.masking import load_probabilities_per_flame_triangle, masking, mesh_based_mask_uniform_faces, transfer_pixels
from model.flame.renderer import Renderer, transform_vertices
from model.generator import UNetGenerator
from model.heads import ComponentHeads
from model.losses.cycle import augment_expression_cycle, expression_cycle_loss, identity_cycle_loss, load_expression_templates
from model.losses.emotion import emotion_loss
from model.losses.landmark import (
    eye_closure_loss,
    fan_boundary_loss,
    landmark_visibility_mask,
    lip_closure_loss,
    mediapipe_landmark_loss,
    mouth_point_indices,
)
from model.losses.mesh import build_gated_expressive_region_weights, build_region_weights
from model.losses.mica_shape import mica_shape_loss
from model.losses.photometric import VGGPerceptualLoss, photometric_loss
from model.losses.temporal_smoothness import compute_vertex_gate, velocity_penalty, vertex_velocity_penalty
from model.temporal import GatedTemporalTransformer, SimpleTemporalTransformer, TemporalTransformer, TTModule
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import Stage2Config, load_stage2_config
from training.distributed import cleanup_distributed, is_distributed, is_main_process, setup_distributed, unwrap_model
from training.eval_loaders import build_eval_loaders
from training.loss_utils import concat_category_fields, gated_loss, next_batch, regularization_loss, weighted_metrics
from training.losses_3d import compute_3d_losses
from training.wandb_utils import flatten_metrics

_REPO_ROOT_RELATIVE_FARL_PATH = "pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"


def _render_and_reconstruct(
    flame: FLAME, renderer: Renderer, unet: UNetGenerator, face_probabilities: torch.Tensor,
    encoded: dict[str, torch.Tensor], pixel_values: torch.Tensor, face_mask: torch.Tensor,
    valid_recon: torch.Tensor,
    precomputed_flame_out: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_for_proj = torch.cat([encoded["scale"], encoded["translation"]], dim=-1)

    flame_out = (
        precomputed_flame_out if precomputed_flame_out is not None
        else flame(encoded["shape"], encoded["expression"], encoded["jaw"], encoded["eyelid"], encoded["rotation"])
    )
    render_out = renderer(
        flame_out["vertices"], cam_for_proj,
        landmarks_fan=flame_out["landmarks_fan"], landmarks_mp=flame_out["landmarks_mp"],
    )
    projected_fan = render_out["transformed_landmarks_fan"]
    projected_mp = render_out["transformed_landmarks_mp"]

    reconstructed = torch.zeros_like(pixel_values)
    if valid_recon.any():
        npoints, _coords = mesh_based_mask_uniform_faces(
            render_out["transformed_vertices"][valid_recon], flame.faces_tensor, face_probabilities,
        )
        extra_points = transfer_pixels(pixel_values[valid_recon], npoints, npoints)
        # masking()'s `mask` is 1=background/keep, 0=face/blackout-candidate -
        # the OPPOSITE polarity from face_parsing_cache's face_mask (XSeg's own
        # convention: 1=visible face skin, 0=background), hence the inversion.
        background_mask = 1 - face_mask[valid_recon].unsqueeze(1)
        masked_img = masking(pixel_values[valid_recon], background_mask, extra_points)

        unet_input = torch.cat([render_out["rendered_img"][valid_recon], masked_img], dim=1)
        reconstructed[valid_recon] = unet(unet_input)
    return reconstructed, projected_fan, projected_mp


def _compute_2d_reconstruction_losses_from_encoded(
    flame: FLAME, renderer: Renderer, unet: UNetGenerator, emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss,
    face_probabilities: torch.Tensor, encoded: dict[str, torch.Tensor], batch_2d: dict[str, torch.Tensor],
    occlusion_mask: torch.Tensor | None = None, occlusion_loss_weight: float = 1.0,
    landmark_occlusion_masking: bool = False,
    precomputed_flame_out: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    valid_recon = batch_2d["flag_face_mask_valid"]
    reconstructed, projected_fan, projected_mp = _render_and_reconstruct(
        flame, renderer, unet, face_probabilities, encoded, batch_2d["pixel_values"], batch_2d["face_mask"],
        valid_recon, precomputed_flame_out=precomputed_flame_out,
    )

    fan_occlusion_mask = mp_occlusion_mask = None
    if landmark_occlusion_masking:
        fan_occlusion_mask = landmark_visibility_mask(batch_2d["face_mask"], batch_2d["landmarks_fan"])
        mp_occlusion_mask = landmark_visibility_mask(batch_2d["face_mask"], batch_2d["landmarks_mp"])

    def _fan_args() -> tuple[torch.Tensor, ...]:
        args = (projected_fan, batch_2d["landmarks_fan"])
        return args + (fan_occlusion_mask,) if landmark_occlusion_masking else args

    def _mp_args() -> tuple[torch.Tensor, ...]:
        args = (projected_mp, batch_2d["landmarks_mp"])
        return args + (mp_occlusion_mask,) if landmark_occlusion_masking else args

    def _aggregate(
        face_mask_valid: torch.Tensor, landmarks_fan_valid: torch.Tensor,
        landmarks_mp_valid: torch.Tensor, mica_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        photometric = gated_loss(photometric_loss, face_mask_valid, reconstructed, batch_2d["pixel_values"])
        vgg = gated_loss(vgg_loss, face_mask_valid, reconstructed, batch_2d["pixel_values"])
        emotion_term = gated_loss(
            lambda r, t: emotion_loss(r, t, emotion_net), face_mask_valid, reconstructed, batch_2d["pixel_values"]
        )
        landmark_loss = gated_loss(
            fan_boundary_loss, landmarks_fan_valid, *_fan_args()
        ) + gated_loss(
            mediapipe_landmark_loss, landmarks_mp_valid, *_mp_args()
        )
        closure_loss = gated_loss(
            eye_closure_loss, landmarks_mp_valid, *_mp_args()
        ) + gated_loss(
            lip_closure_loss, landmarks_mp_valid, *_mp_args()
        )
        mica_loss = gated_loss(mica_shape_loss, mica_valid, encoded["shape"], batch_2d["mica_shape"])
        loss_excl_emotion = (
            constants.PHOTOMETRIC_LOSS_WEIGHT * photometric
            + constants.VGG_LOSS_WEIGHT * vgg
            + constants.LANDMARK_LOSS_WEIGHT * landmark_loss
            + constants.CLOSURE_LOSS_WEIGHT * closure_loss
            + constants.MICA_SHAPE_LOSS_WEIGHT * mica_loss
        )
        term_metrics = {
            "photometric": photometric.item(), "vgg": vgg.item(), "emotion": emotion_term.item(),
            "landmark": landmark_loss.item(), "closure": closure_loss.item(), "mica": mica_loss.item(),
        }
        return loss_excl_emotion, emotion_term, term_metrics

    loss_excluding_emotion, emotion_term, metrics = _aggregate(
        batch_2d["flag_face_mask_valid"], batch_2d["flag_landmarks_fan_valid"],
        batch_2d["flag_landmarks_mp_valid"], batch_2d["flag_mica_valid"],
    )
    if occlusion_mask is not None and occlusion_loss_weight != 1.0:
        extra_loss_excl_emotion, extra_emotion_term, _ = _aggregate(
            occlusion_mask & batch_2d["flag_face_mask_valid"], occlusion_mask & batch_2d["flag_landmarks_fan_valid"],
            occlusion_mask & batch_2d["flag_landmarks_mp_valid"], occlusion_mask & batch_2d["flag_mica_valid"],
        )
        extra_weight = occlusion_loss_weight - 1.0
        loss_excluding_emotion = loss_excluding_emotion + extra_weight * extra_loss_excl_emotion
        emotion_term = emotion_term + extra_weight * extra_emotion_term

    reg_loss = regularization_loss(encoded)
    loss_excluding_emotion = loss_excluding_emotion + reg_loss
    metrics["reg_2d"] = reg_loss.item()
    return loss_excluding_emotion, emotion_term, metrics


def compute_2d_reconstruction_losses(
    svit: SViT, heads: ComponentHeads, flame: FLAME, renderer: Renderer, unet: UNetGenerator,
    emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss, face_probabilities: torch.Tensor,
    batch_2d: dict[str, torch.Tensor],
    landmark_occlusion_masking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    encoded = encode_image(svit, heads, batch_2d["pixel_values"])
    return _compute_2d_reconstruction_losses_from_encoded(
        flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, encoded, batch_2d,
        landmark_occlusion_masking=landmark_occlusion_masking,
    )


def run_pass_a(
    svit: SViT, heads: ComponentHeads, flame: FLAME, renderer: Renderer, unet: UNetGenerator,
    emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss, region_weights: torch.Tensor,
    face_probabilities: torch.Tensor, optimizer: torch.optim.Optimizer,
    batch_2d: dict[str, torch.Tensor], batch_3d: dict[str, torch.Tensor], subject_ids_3d: list[str], device: str,
    freeze_encoder: bool = False,
    landmark_occlusion_masking: bool = False,
) -> dict[str, float]:
    for p in list(svit.parameters()) + list(heads.parameters()):
        p.requires_grad_(not freeze_encoder)
    for p in unet.parameters():
        p.requires_grad_(True)

    optimizer.zero_grad()

    # 2D branch first: forward, then BOTH its backwards (main + the emotion
    # carve-out), fully completed before the 3D branch's forward is even
    # built - keeps only one branch's reconstruction graph (Renderer/UNet/VGG
    # activations, held open across both backwards by retain_graph=True)
    # resident in memory at a time, rather than building both the 2D and 3D
    # forward graphs simultaneously before either backward runs. Confirmed via
    # a real GPU OOM at production batch size: Pass A's 2D branch alone (full
    # photometric+VGG+UNet reconstruction, retained for the emotion double-
    # backward) is categorically heavier than Stage 1's equivalent, which
    # never needed retain_graph at all - see training/pretrain.py's
    # compute_2d_losses. Splitting the backward this way changes nothing
    # about what gets optimized: backward() accumulates into .grad rather
    # than overwriting it, so two sequential backward() calls before one
    # optimizer.step() produce the exact same accumulated gradient as one
    # combined backward on the summed loss - only peak memory differs.
    loss_2d_excl_emotion, emotion_term, metrics_2d = compute_2d_reconstruction_losses(
        svit, heads, flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, batch_2d,
        landmark_occlusion_masking=landmark_occlusion_masking,
    )
    loss_2d_scaled = constants.LOSS_BALANCE_2D * loss_2d_excl_emotion
    emotion_loss_scaled = constants.LOSS_BALANCE_2D * constants.EMOTION_LOSS_WEIGHT * emotion_term

    # retain_graph=True: the UNet's output tensor (and everything upstream of
    # it - UNet, FLAME, encoder) is reused by emotion_loss_scaled's backward
    # below, so the graph can't be freed after this first call.
    #
    # requires_grad guard: normally unconditional - loss_2d_excl_emotion's own
    # reg_loss term (regularization_loss(encoded)) is ungated by valid_recon,
    # so it's always grad-connected to svit/heads regardless of batch
    # validity, guaranteeing loss_2d_scaled has SOME trainable-parameter path
    # even in the pathological all-invalid-batch case. With freeze_encoder,
    # that guarantee is gone (reg_loss's only path was svit/heads, now
    # frozen) - the sole remaining path is photometric/vgg through unet,
    # which itself is zero/disconnected when valid_recon.any() is False (see
    # _render_and_reconstruct: reconstructed stays all-zero, unet never
    # runs). Same disconnected-tensor crash as loss_3d_scaled below in that
    # combination (rare - needs a fully-invalid 2D batch - but possible, and
    # this function no longer has a structural guarantee against it).
    if loss_2d_scaled.requires_grad:
        loss_2d_scaled.backward(retain_graph=True)

    unet_params = list(unet.parameters())
    for p in unet_params:
        p.requires_grad_(False)
    # emotion_term is gated by flag_face_mask_valid - if EVERY sample in
    # batch_2d happens to be invalid, it's a disconnected zero tensor with no
    # grad_fn (same edge case already handled in run_pass_b), and calling
    # backward() on it would crash - same guard, same reason as
    # loss_2d_scaled's own guard above.
    if emotion_loss_scaled.requires_grad:
        emotion_loss_scaled.backward()
    for p in unet_params:
        p.requires_grad_(True)

    # 3D branch: forward + backward only now, after the 2D branch's graph has
    # been fully backpropped (both calls above) and freed. compute_3d_losses'
    # own regularization term is likewise always ungated/grad-connected
    # (training/losses_3d.py) UNLESS freeze_encoder - mesh/Lvc/reg_3d only
    # ever touch svit/heads (3D batches get direct 3D vertex supervision, no
    # renderer/unet in that path at all - training/losses_3d.py's
    # compute_3d_losses calls encode_image+flame only), so with the encoder
    # frozen loss_3d_scaled has no trainable-parameter connection whatsoever -
    # a fully disconnected tensor, and backward() on that crashes (confirmed:
    # "RuntimeError: element 0 of tensors does not require grad and does not
    # have a grad_fn" from exactly this call, the first real run of a
    # freeze_encoder=True config). Same requires_grad guard as the emotion
    # carve-out above, same reason.
    loss_3d, metrics_3d = compute_3d_losses(svit, heads, flame, region_weights, batch_3d, subject_ids_3d, device)
    loss_3d_scaled = constants.LOSS_BALANCE_3D * loss_3d
    if loss_3d_scaled.requires_grad:
        loss_3d_scaled.backward()

    optimizer.step()

    total = loss_2d_scaled.item() + emotion_loss_scaled.item() + loss_3d_scaled.item()
    metrics = {"total": total, "2d": metrics_2d, "3d": metrics_3d}
    return metrics


def compute_cycle_losses(
    svit: SViT, heads: ComponentHeads, flame: FLAME, renderer: Renderer, unet: UNetGenerator,
    face_probabilities: torch.Tensor, templates: dict, batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = encode_image(svit, heads, batch["pixel_values"])
    cam_for_proj = torch.cat([encoded["scale"], encoded["translation"]], dim=-1)

    aug_expression, aug_jaw, aug_eyelid = augment_expression_cycle(
        encoded["expression"], encoded["jaw"], encoded["eyelid"], templates,
    )

    valid_recon = batch["flag_face_mask_valid"]
    reconstructed = torch.zeros_like(batch["pixel_values"])
    if valid_recon.any():
        encoded_v = {k: v[valid_recon] for k, v in encoded.items()}
        cam_for_proj_v = cam_for_proj[valid_recon]

        flame_out = flame(
            encoded_v["shape"], encoded_v["expression"], encoded_v["jaw"], encoded_v["eyelid"],
            encoded_v["rotation"],
        )
        # Only transformed_vertices is needed from the pre-augmentation pose (to
        # sample which mesh points/triangles to track) - transform_vertices avoids
        # the full Renderer's (wasted) rasterization of an image that's never used,
        # since only the post-augmentation pose's rendered_img feeds the UNet below.
        transformed_vertices = transform_vertices(flame_out["vertices"], cam_for_proj_v)
        npoints1, coords = mesh_based_mask_uniform_faces(transformed_vertices, flame.faces_tensor, face_probabilities)

        aug_expression_v, aug_jaw_v, aug_eyelid_v = (
            aug_expression[valid_recon], aug_jaw[valid_recon], aug_eyelid[valid_recon],
        )

        flame_out_aug = flame(encoded_v["shape"], aug_expression_v, aug_jaw_v, aug_eyelid_v, encoded_v["rotation"])
        render_out_aug = renderer(flame_out_aug["vertices"], cam_for_proj_v)
        # coords=coords: resamples the SAME mesh points/triangles npoints1 came
        # from, now re-projected after the augmented re-pose - required for
        # transfer_pixels below to know where each real pixel value should move to.
        npoints2, _coords = mesh_based_mask_uniform_faces(
            render_out_aug["transformed_vertices"], flame.faces_tensor, face_probabilities, coords=coords,
        )

        extra_points = transfer_pixels(batch["pixel_values"][valid_recon], npoints1, npoints2)
        # Same mask-polarity inversion as Pass A (masking()'s mask is 1=background/
        # keep, 0=face/blackout - the opposite of face_parsing_cache's face_mask).
        background_mask = 1 - batch["face_mask"][valid_recon].unsqueeze(1)
        masked_img = masking(batch["pixel_values"][valid_recon], background_mask, extra_points)

        unet_input = torch.cat([render_out_aug["rendered_img"], masked_img], dim=1)
        reconstructed[valid_recon] = unet(unet_input)

    re_encoded = encode_image(svit, heads, reconstructed)

    # Gated by flag_face_mask_valid (same reasoning as Pass A): a no-face
    # sample's face_mask fallback is all-zero, making background_mask above
    # all-ones - masking() then blacks out nothing at all, so masked_img is
    # essentially the unmodified original image, and the "reconstruction"
    # degenerates into copying the input back out rather than the intended
    # geometry+sparse-pixels task. Re-encoding that degenerate reconstruction
    # would otherwise feed a meaningless value into these losses.
    expr_cycle = gated_loss(
        expression_cycle_loss, valid_recon,
        re_encoded["expression"], aug_expression, re_encoded["jaw"], aug_jaw, re_encoded["eyelid"], aug_eyelid,
    )
    # Compared against the ORIGINAL (pre-augmentation) shape, not the augmented
    # target - shape is never itself augmented (see augment_expression_cycle's
    # docstring), only expression/jaw/eyelid are.
    id_cycle = gated_loss(identity_cycle_loss, valid_recon, re_encoded["shape"], encoded["shape"])
    # Regularization on the RE-ENCODED params, not the first pass's: this is
    # what this pass's gradient actually flows through and updates.
    # regularization_loss takes a dict (not raw tensors), so gated_loss's
    # *tensors signature doesn't directly apply - filtered manually instead.
    if valid_recon.any():
        reg_loss = regularization_loss({k: re_encoded[k][valid_recon] for k in ("shape", "expression", "jaw")})
    else:
        reg_loss = torch.zeros((), device=batch["pixel_values"].device)

    total = constants.CYCLE_LOSS_WEIGHT * expr_cycle + constants.IDENTITY_CYCLE_LOSS_WEIGHT * id_cycle + reg_loss
    metrics = {"expr_cycle": expr_cycle.item(), "id_cycle": id_cycle.item(), "reg": reg_loss.item()}
    return total, metrics


def run_pass_b(
    svit: SViT, heads: ComponentHeads, flame: FLAME, renderer: Renderer, unet: UNetGenerator,
    face_probabilities: torch.Tensor, templates: dict, optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor], mode: Literal["encoder", "unet", "joint"],
) -> dict[str, float]:
    for p in list(svit.parameters()) + list(heads.parameters()):
        p.requires_grad_(mode in ("encoder", "joint"))
    for p in unet.parameters():
        p.requires_grad_(mode in ("unet", "joint"))

    total_loss, metrics = compute_cycle_losses(svit, heads, flame, renderer, unet, face_probabilities, templates, batch)

    optimizer.zero_grad()
    # All three of compute_cycle_losses' terms are now gated by
    # flag_face_mask_valid - if EVERY sample in this batch happens to be
    # invalid (astronomically unlikely at real batch sizes, but not
    # impossible for a tiny test batch), total_loss is a disconnected zero
    # tensor with no grad_fn at all, and calling backward() on it would crash.
    # Skip the step gracefully in that case rather than erroring.
    if total_loss.requires_grad:
        total_loss.backward()
        optimizer.step()

    metrics["total"] = total_loss.item()
    return metrics


def _flatten(t: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    """(B, N, ...) -> (B*N, ...), mirroring encode_video's own SViT-flattening
    approach - used to feed encode_video's per-frame output (and the matching
    per-frame batch fields) into loss functions that expect a flat batch dim."""
    return t.reshape(batch_size * num_frames, *t.shape[2:])


def compute_2d_video_losses(
    flame: FLAME, renderer: Renderer, unet: UNetGenerator, emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss,
    face_probabilities: torch.Tensor, encoded: dict[str, torch.Tensor],
    batch_2d_video: dict[str, torch.Tensor], real_frame_mask: torch.Tensor,
    occlusion_mask: torch.Tensor | None = None, occlusion_loss_weight: float = 1.0,
    landmark_occlusion_masking: bool = False,
    precomputed_flame_out: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, num_frames = real_frame_mask.shape
    encoded_flat = {k: _flatten(v, batch_size, num_frames) for k, v in encoded.items()}
    batch_flat = {k: _flatten(v, batch_size, num_frames) for k, v in batch_2d_video.items()}
    real_frame_mask_flat = _flatten(real_frame_mask, batch_size, num_frames)
    occlusion_mask_flat = _flatten(occlusion_mask, batch_size, num_frames) if occlusion_mask is not None else None

    for flag_key in ("flag_face_mask_valid", "flag_landmarks_fan_valid", "flag_landmarks_mp_valid", "flag_mica_valid"):
        batch_flat[flag_key] = real_frame_mask_flat & batch_flat[flag_key]

    if precomputed_flame_out is not None:
        assert precomputed_flame_out["vertices"].shape[0] == encoded_flat["shape"].shape[0], (
            "precomputed_flame_out row count doesn't match this function's own flatten of `encoded` - "
            "row-order invariant broken (see this function's own docstring)"
        )

    loss_excluding_emotion, emotion_term, metrics = _compute_2d_reconstruction_losses_from_encoded(
        flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, encoded_flat, batch_flat,
        occlusion_mask_flat, occlusion_loss_weight,
        landmark_occlusion_masking=landmark_occlusion_masking, precomputed_flame_out=precomputed_flame_out,
    )
    total = loss_excluding_emotion + constants.EMOTION_LOSS_WEIGHT * emotion_term
    return total, metrics


def compute_temporal_smoothness_losses(
    encoded: dict[str, torch.Tensor], real_frame_mask: torch.Tensor,
    occlusion_mask: torch.Tensor | None = None, occlusion_loss_weight: float = 1.0,
    param_smoothness_enabled: bool = True,
    vertices: torch.Tensor | None = None,
    base_region_weights: torch.Tensor | None = None,
    gated_region_mask: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    expressive_region_smooth_weight: float = 3.0,
    temporal_vertex_smoothness_weight: float = 0.0,
    temporal_velocity_weight: float = constants.TEMPORAL_VELOCITY_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.zeros((), device=real_frame_mask.device)
    metrics: dict[str, float] = {}

    if param_smoothness_enabled:
        expr_eyelid = torch.cat([encoded["expression"], encoded["eyelid"]], dim=-1)
        camera_rotation = torch.cat([encoded["scale"], encoded["rotation"]], dim=-1)

        frame_weight = None
        if occlusion_mask is not None and occlusion_loss_weight != 1.0:
            frame_weight = torch.where(
                occlusion_mask, occlusion_loss_weight, torch.ones((), device=occlusion_mask.device)
            )

        vel_expr = velocity_penalty(expr_eyelid, real_frame_mask, frame_weight)
        vel_jaw = velocity_penalty(encoded["jaw"], real_frame_mask, frame_weight)
        vel_camera = velocity_penalty(camera_rotation, real_frame_mask, frame_weight)
        vel_shape = velocity_penalty(encoded["shape"], real_frame_mask, frame_weight)

        param_term = temporal_velocity_weight * (vel_expr + vel_jaw + vel_camera + vel_shape)
        total = total + param_term
        metrics.update({
            "vel_expr": vel_expr.item(), "vel_jaw": vel_jaw.item(),
            "vel_camera": vel_camera.item(), "vel_shape": vel_shape.item(),
        })

    if temporal_vertex_smoothness_weight > 0:
        assert vertices is not None and base_region_weights is not None, (
            "vertices/base_region_weights required when temporal_vertex_smoothness_weight > 0"
        )
        assert gated_region_mask is not None and gate is not None, (
            "gated_region_mask/gate required when temporal_vertex_smoothness_weight > 0"
        )
        vertex_term = vertex_velocity_penalty(
            vertices, base_region_weights, gated_region_mask, gate, expressive_region_smooth_weight,
            valid_mask=real_frame_mask,
        )
        total = total + temporal_vertex_smoothness_weight * vertex_term
        metrics["vertex_smooth"] = vertex_term.item()

    return total, metrics


def run_pass_c(
    svit: SViT, tt: TTModule, heads: ComponentHeads, flame: FLAME, renderer: Renderer,
    unet: UNetGenerator, emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss,
    face_probabilities: torch.Tensor, optimizer: torch.optim.Optimizer,
    batch_2d_video: dict[str, torch.Tensor],
    synthetic_occlusion_enabled: bool = False, synthetic_occlusion_prob: float = 0.0,
    occlusion_loss_weight: float = 1.0,
    landmark_occlusion_masking: bool = False,
    base_region_weights: torch.Tensor | None = None,
    gated_region_mask: torch.Tensor | None = None,
    expressive_region_smooth_weight: float = 3.0,
    temporal_vertex_smoothness_weight: float = 0.0,
    param_smoothness_in_pass_c: bool = True,
    mouth_gate_use_region_visibility: bool = False,
    identity_pooling: bool = False,
    vertex_gate_mode: str = "min_vis",
    vertex_gate_delta_cap: float = 0.1,
    vertex_gate_delta_beta: float = 1.0,
    temporal_velocity_weight_in_pass_c: float = constants.TEMPORAL_VELOCITY_WEIGHT,
) -> dict[str, float]:
    for p in list(svit.parameters()) + list(heads.parameters()) + list(unet.parameters()):
        p.requires_grad_(False)
    for p in tt.parameters():
        p.requires_grad_(True)

    num_frames = batch_2d_video["pixel_values"].shape[1]
    device = batch_2d_video["pixel_values"].device
    batch_size = batch_2d_video["pixel_values"].shape[0]
    frame_indices = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1)

    occlusion_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=device)
    if synthetic_occlusion_enabled and synthetic_occlusion_prob > 0:
        eligible = batch_2d_video["valid_mask"] & batch_2d_video["flag_visibility_valid"]
        occlusion_mask = eligible & (torch.rand(batch_size, num_frames, device=device) < synthetic_occlusion_prob)

    visibility_for_tt = batch_2d_video["visibility_ratio"].masked_fill(occlusion_mask, 0.0)
    flag_visibility_for_tt = batch_2d_video["flag_visibility_valid"] & ~occlusion_mask

    # Change 2's gate signal, computed before encode_video (doesn't depend on
    # it) - only when the vertex-space term is actually active, to avoid
    # wasted compute (mirrors the synthetic_occlusion_enabled guard above).
    gate = None
    if temporal_vertex_smoothness_weight > 0:
        if mouth_gate_use_region_visibility:
            face_mask_flat = _flatten(batch_2d_video["face_mask"], batch_size, num_frames)
            landmarks_mp_flat = _flatten(batch_2d_video["landmarks_mp"], batch_size, num_frames)
            point_visible = landmark_visibility_mask(face_mask_flat, landmarks_mp_flat)  # (B*N, 105)
            lip_idx = mouth_point_indices().to(point_visible.device)
            mouth_vis_flat = point_visible[:, lip_idx].float().mean(dim=-1)  # (B*N,)
            gate_signal = mouth_vis_flat.reshape(batch_size, num_frames)
        else:
            gate_signal = visibility_for_tt  # whole-face score, already zeroes synthetic-occlusion positions
        gate = compute_vertex_gate(
            gate_signal, mode=vertex_gate_mode, cap=vertex_gate_delta_cap, beta=vertex_gate_delta_beta,
        )  # (B, N-1)

    encoded_2d = encode_video(
        svit, tt, heads, batch_2d_video["pixel_values"], visibility_for_tt,
        frame_indices, flag_visibility_for_tt, batch_2d_video["valid_mask"],
        pool_identity=identity_pooling,
    )
    real_frame_mask_2d = batch_2d_video["valid_mask"]

    optimizer.zero_grad()

    # FLAME forward once, before either backward - see docstring above.
    encoded_flat = {k: _flatten(v, batch_size, num_frames) for k, v in encoded_2d.items()}
    flame_out_flat = flame(
        encoded_flat["shape"], encoded_flat["expression"], encoded_flat["jaw"],
        encoded_flat["eyelid"], encoded_flat["rotation"],
    )
    vertices_for_temporal = None
    if temporal_vertex_smoothness_weight > 0:
        vertices_for_temporal = flame_out_flat["vertices"].reshape(batch_size, num_frames, -1, 3)

    loss_temporal, metrics_temporal = compute_temporal_smoothness_losses(
        encoded_2d, real_frame_mask_2d, occlusion_mask, occlusion_loss_weight,
        param_smoothness_enabled=param_smoothness_in_pass_c,
        vertices=vertices_for_temporal, base_region_weights=base_region_weights,
        gated_region_mask=gated_region_mask, gate=gate,
        expressive_region_smooth_weight=expressive_region_smooth_weight,
        temporal_vertex_smoothness_weight=temporal_vertex_smoothness_weight,
        temporal_velocity_weight=temporal_velocity_weight_in_pass_c,
    )
    # retain_graph=True: flame_out_flat's graph (now built BEFORE this call,
    # unlike before the FLAME-once restructure) is still needed by loss_2d's
    # backward below - cheap to retain here, since the heavy reconstruction
    # graph (UNet/VGG) hasn't been built yet.
    loss_temporal.backward(retain_graph=True)

    loss_2d, metrics_2d = compute_2d_video_losses(
        flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, encoded_2d, batch_2d_video, real_frame_mask_2d,
        occlusion_mask, occlusion_loss_weight,
        landmark_occlusion_masking=landmark_occlusion_masking, precomputed_flame_out=flame_out_flat,
    )
    loss_2d.backward()

    optimizer.step()

    total_loss = loss_2d.item() + loss_temporal.item()
    metrics = {"total": total_loss, "2d": metrics_2d, "temporal": metrics_temporal}
    return metrics


@torch.no_grad()
def run_periodic_eval_local(
    svit: SViT, tt: TTModule, heads: ComponentHeads, flame: FLAME, renderer: Renderer,
    eval_loaders: dict[str, DataLoader], device: str,
    pool_identity: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    svit, tt, heads = unwrap_model(svit), unwrap_model(tt), unwrap_model(heads)

    results: dict[str, dict[str, np.ndarray]] = {}
    for dataset_name, loader in eval_loaders.items():
        landmark_fan_errors: list[float] = []
        landmark_mp_errors: list[float] = []
        temporal_errors: list[float] = []

        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            batch_size, num_frames = pixel_values.shape[:2]
            frame_indices = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1)
            encoded = encode_video(
                svit, tt, heads, pixel_values, batch["visibility_ratio"].to(device),
                frame_indices, batch["flag_visibility_valid"].to(device), batch["valid_mask"].to(device),
                pool_identity=pool_identity,
            )
            encoded_flat = {k: _flatten(v, batch_size, num_frames) for k, v in encoded.items()}

            cam_for_proj = torch.cat([encoded_flat["scale"], encoded_flat["translation"]], dim=-1)
            flame_out = flame(
                encoded_flat["shape"], encoded_flat["expression"], encoded_flat["jaw"],
                encoded_flat["eyelid"], encoded_flat["rotation"],
            )
            render_out = renderer(
                flame_out["vertices"], cam_for_proj,
                landmarks_fan=flame_out["landmarks_fan"], landmarks_mp=flame_out["landmarks_mp"],
            )
            projected_fan = render_out["transformed_landmarks_fan"].cpu().numpy()
            projected_mp = render_out["transformed_landmarks_mp"].cpu().numpy()

            # real_frame_mask excludes tail padding (see compute_2d_video_losses'
            # own identical use of this AND pattern), on top of each field's own
            # per-frame detection-validity flag.
            real_frame_mask_flat = _flatten(batch["valid_mask"], batch_size, num_frames).numpy()
            gt_fan = _flatten(batch["landmarks_fan_full"], batch_size, num_frames).numpy()
            flag_fan = _flatten(batch["flag_landmarks_fan_full_valid"], batch_size, num_frames).numpy() & real_frame_mask_flat
            gt_mp = _flatten(batch["landmarks_mp"], batch_size, num_frames).numpy()
            flag_mp = _flatten(batch["flag_landmarks_mp_valid"], batch_size, num_frames).numpy() & real_frame_mask_flat

            for i in range(projected_fan.shape[0]):
                landmark_fan_errors.append(
                    per_frame_euclidean_error(projected_fan[i], gt_fan[i]) if flag_fan[i] else np.nan
                )
                landmark_mp_errors.append(
                    per_frame_euclidean_error(projected_mp[i], gt_mp[i]) if flag_mp[i] else np.nan
                )

            vertices = flame_out["vertices"].reshape(batch_size, num_frames, *flame_out["vertices"].shape[1:])
            vertices = vertices.cpu().numpy()
            valid_mask_np = batch["valid_mask"].numpy()
            for b in range(batch_size):
                for t in range(1, num_frames):
                    if valid_mask_np[b, t - 1] and valid_mask_np[b, t]:
                        temporal_errors.append(per_frame_vertex_error(vertices[b, t - 1], vertices[b, t]))
                    else:
                        temporal_errors.append(np.nan)

        results[dataset_name] = {
            "landmark_fan": np.array(landmark_fan_errors, dtype=np.float64),
            "landmark_mp": np.array(landmark_mp_errors, dtype=np.float64),
            "temporal_smoothness": np.array(temporal_errors, dtype=np.float64),
        }
    return results


def aggregate_and_print_eval_results(
    local_results: dict[str, dict[str, np.ndarray]], rank: int, world_size: int, step: int,
) -> None:
    if is_distributed():
        gathered: list[dict[str, dict[str, np.ndarray]]] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
        if rank != 0:
            return
        all_results = gathered
    else:
        all_results = [local_results]

    all_summaries: dict[str, float] = {}
    for dataset_name in all_results[0]:
        summaries = {
            metric_name: summarize(np.concatenate([r[dataset_name][metric_name] for r in all_results]))
            for metric_name in all_results[0][dataset_name]
        }
        print(f"step {step} eval[{dataset_name}]: {summaries}")
        means_only = {metric_name: stats["mean"] for metric_name, stats in summaries.items()}
        all_summaries.update(flatten_metrics(means_only, prefix=f"eval/{dataset_name}"))

    if wandb.run is not None:
        wandb.log(all_summaries, step=step)


def train(cfg: Stage2Config, checkpoint_pth: str | None = None) -> None:
    torch.manual_seed(cfg.seed)
    rank, world_size, local_rank, device = setup_distributed(fallback_device=cfg.device)
    if is_main_process(rank):
        print(f"config: {cfg}")
        wandb.init(
            project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.wandb_run_name,
            job_type="stage2", config=dataclasses.asdict(cfg),
        )

    try:
        svit = SViT().to(device)
        load_farl_pretrained(svit, _REPO_ROOT_RELATIVE_FARL_PATH)
        heads = ComponentHeads(expression_dim=cfg.num_expression_params).to(device)
        flame = FLAME(n_exp=cfg.num_expression_params).to(device)
        renderer = Renderer(flame.faces_tensor).to(device)
        unet = UNetGenerator().to(device)
        if cfg.tt_variant == "simple":
            tt = SimpleTemporalTransformer().to(device)
        elif cfg.tt_variant == "gated":
            tt = GatedTemporalTransformer(GatedTTConfig(gamma=cfg.tt_gamma)).to(device)
        else:
            tt = TemporalTransformer().to(device)
        emotion_net = EmotionNet().to(device)
        vgg_loss = VGGPerceptualLoss().to(device)
        region_weights = build_region_weights().to(device)
        base_region_weights, gated_region_mask = build_gated_expressive_region_weights()
        base_region_weights, gated_region_mask = base_region_weights.to(device), gated_region_mask.to(device)
        face_probabilities = load_probabilities_per_flame_triangle().to(device)
        templates = load_expression_templates()

        # One shared optimizer over all four modules - each run_pass_*
        # function gates which subset actually moves via its own
        # requires_grad_ toggling (see their own docstrings), even though the
        # optimizer's own param list spans everyone.
        optimizer = torch.optim.Adam(
            list(svit.parameters()) + list(heads.parameters()) + list(unet.parameters()) + list(tt.parameters()),
            lr=cfg.learning_rate,
        )

        start_step = 0
        if checkpoint_pth is not None:
            # A non-"original" tt_variant has different q_proj/k_proj shapes than
            # the original TT (model/temporal.py) - load_checkpoint's strict=False
            # only tolerates missing/unexpected KEYS, not shape mismatches on a
            # shared key name, so passing "tt" here would crash rather than
            # silently skip it. svit/heads/unet still warm-start from
            # checkpoint_pth; tt itself just starts fresh (identity-at-init, same
            # as any new run - see TemporalTransformer's docstring).
            assert cfg.tt_variant == "original" or not cfg.resume_optimizer_and_step, (
                "resume_optimizer_and_step=True together with a non-'original' tt_variant "
                "would try to load the old tt's optimizer state into a differently-shaped "
                "tt - set resume_optimizer_and_step=False when changing tt_variant"
            )
            modules_to_load = {"svit": svit, "heads": heads, "unet": unet}
            if cfg.tt_variant == "original":
                modules_to_load["tt"] = tt
            if cfg.resume_optimizer_and_step:
                loaded_step = load_checkpoint(
                    checkpoint_pth, modules_to_load, optimizer, device,
                    expected_num_expression_params=cfg.num_expression_params,
                )
                start_step = loaded_step + 1
            else:
                load_checkpoint(
                    checkpoint_pth, modules_to_load, device=device,
                    expected_num_expression_params=cfg.num_expression_params,
                )
        else:
            # One-time seed from Stage 1 - svit+heads only, no optimizer, no
            # unet/tt (Stage 1 never saved any) - only on a fresh run (no
            # Stage-2-own checkpoint to resume from). See Stage2Config's own
            # docstring for why this is skipped on a resume.
            load_checkpoint(
                cfg.stage1_checkpoint_pth, {"svit": svit, "heads": heads},
                expected_num_expression_params=cfg.num_expression_params,
            )

        if is_distributed():
            # find_unused_parameters=True: Stage 2's per-pass requires_grad_
            # toggling means different subsets of these four modules are
            # "live" on different calls - DDP's hooks are registered once at
            # construction based on all parameters that existed then, and by
            # default expect gradients for all of them on every backward();
            # without this flag, a step where some parameters get no gradient
            # (because that pass froze them) would hang or error under
            # multi-GPU training. Stage 1 never needed this - svit/heads were
            # always fully trainable there, no per-step freezing at all.
            # device_ids=[0], not [local_rank]: setup_distributed now restricts
            # CUDA_VISIBLE_DEVICES to exactly this rank's own GPU, so every
            # process's own device is always index 0 in its own restricted view,
            # regardless of local_rank's original torchrun-assigned value.
            svit = DistributedDataParallel(svit, device_ids=[0], find_unused_parameters=True)
            heads = DistributedDataParallel(heads, device_ids=[0], find_unused_parameters=True)
            unet = DistributedDataParallel(unet, device_ids=[0], find_unused_parameters=True)
            tt = DistributedDataParallel(tt, device_ids=[0], find_unused_parameters=True)

        dataloader_cfg = load_dataloader_config(cfg.dataloader_config_path)
        frame_pool_loader = build_combined_loader(
            dataloader_cfg, split="train", rank=rank, world_size=world_size,
            datasets_yaml_path=cfg.datasets_yaml_path, video_mode="frame_pool",
        )
        clip_loader = build_combined_loader(
            dataloader_cfg, split="train", rank=rank, world_size=world_size,
            datasets_yaml_path=cfg.datasets_yaml_path, video_mode="clip",
            occlusion_index_dir=cfg.pass_c_occlusion_subset_index_dir,
        )

        # On a resume, start each loader's own epoch counter from an
        # approximation (step // len(loader)) rather than 0 - avoids reusing
        # the exact same early shuffle orders after a resume (training/
        # checkpoint.py's save_checkpoint docstring has the full reasoning).
        # Each loader gets its OWN independent epoch/iterator state.
        frame_pool_epoch = start_step // len(frame_pool_loader)
        frame_pool_loader.set_epoch(frame_pool_epoch)
        frame_pool_iterator = iter(frame_pool_loader)

        clip_epoch = start_step // len(clip_loader)
        clip_loader.set_epoch(clip_epoch)
        clip_iterator = iter(clip_loader)

        # Built once per rank, not per eval round (see training/eval_loaders.py) -
        # every rank builds its own shard, since periodic eval below runs on
        # every rank in parallel (not rank-0-only), gathering results at the end.
        eval_loaders = None
        if cfg.eval.interval_steps > 0:
            eval_loaders = build_eval_loaders(
                dataloader_cfg, cfg.datasets_yaml_path, cfg.eval.datasets,
                cfg.eval.num_clips_per_dataset, rank, world_size,
            )

        keys_2d_a = [
            "pixel_values", "face_mask", "flag_face_mask_valid",
            "mica_shape", "flag_mica_valid",
            "landmarks_fan", "flag_landmarks_fan_valid", "landmarks_mp", "flag_landmarks_mp_valid",
        ]
        keys_3d_a = ["pixel_values", "flame_vertices"]
        keys_b = ["pixel_values", "face_mask", "flag_face_mask_valid"]
        keys_2d_video = [
            "pixel_values", "face_mask", "flag_face_mask_valid",
            "mica_shape", "flag_mica_valid",
            "landmarks_fan", "flag_landmarks_fan_valid", "landmarks_mp", "flag_landmarks_mp_valid",
            "visibility_ratio", "flag_visibility_valid", "valid_mask",
        ]
        pass_b_call_count = 0
        step = start_step - 1  # in case cfg.num_steps <= start_step, so the final-checkpoint save below still has a defined step

        for step in range(start_step, cfg.num_steps):
            if cfg.warmup_steps > 0:
                lr_scale = min(1.0, (step + 1) / cfg.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = cfg.learning_rate * lr_scale

            pass_type = cfg.pass_pattern[step % len(cfg.pass_pattern)]

            if pass_type == "A":
                batch, frame_pool_iterator, frame_pool_epoch = next_batch(frame_pool_loader, frame_pool_iterator, frame_pool_epoch)
                batch_2d = concat_category_fields(batch, ["2d_image", "2d_video"], keys_2d_a, device)
                batch_3d = concat_category_fields(batch, ["3d_image"], keys_3d_a, device)
                subject_ids_3d = batch["3d_image"]["subject_id"]
                metrics = run_pass_a(
                    svit, heads, flame, renderer, unet, emotion_net, vgg_loss, region_weights,
                    face_probabilities, optimizer, batch_2d, batch_3d, subject_ids_3d, device,
                    freeze_encoder=cfg.freeze_encoder,
                    landmark_occlusion_masking=cfg.landmark_occlusion_masking,
                )
            elif pass_type == "B":
                batch, frame_pool_iterator, frame_pool_epoch = next_batch(frame_pool_loader, frame_pool_iterator, frame_pool_epoch)
                batch_b = concat_category_fields(batch, ["2d_image", "2d_video", "3d_image"], keys_b, device)
                cycle_pos = pass_b_call_count % (cfg.pass_b_encoder_steps + cfg.pass_b_unet_steps + cfg.pass_b_joint_steps)
                if cycle_pos < cfg.pass_b_encoder_steps:
                    pass_b_mode = "encoder"
                elif cycle_pos < cfg.pass_b_encoder_steps + cfg.pass_b_unet_steps:
                    pass_b_mode = "unet"
                else:
                    pass_b_mode = "joint"
                metrics = run_pass_b(
                    svit, heads, flame, renderer, unet, face_probabilities, templates, optimizer, batch_b, pass_b_mode,
                )
                pass_b_call_count += 1
            elif pass_type == "C":
                batch, clip_iterator, clip_epoch = next_batch(clip_loader, clip_iterator, clip_epoch)
                batch_2d_video = {k: batch["2d_video"][k].to(device) for k in keys_2d_video}
                metrics = run_pass_c(
                    svit, tt, heads, flame, renderer, unet, emotion_net, vgg_loss,
                    face_probabilities, optimizer, batch_2d_video,
                    synthetic_occlusion_enabled=cfg.pass_c_synthetic_occlusion_enabled,
                    synthetic_occlusion_prob=cfg.pass_c_synthetic_occlusion_prob,
                    occlusion_loss_weight=cfg.pass_c_occlusion_loss_weight,
                    landmark_occlusion_masking=cfg.landmark_occlusion_masking,
                    base_region_weights=base_region_weights, gated_region_mask=gated_region_mask,
                    expressive_region_smooth_weight=cfg.expressive_region_smooth_weight,
                    temporal_vertex_smoothness_weight=cfg.temporal_vertex_smoothness_weight,
                    param_smoothness_in_pass_c=cfg.param_smoothness_in_pass_c,
                    mouth_gate_use_region_visibility=cfg.mouth_gate_use_region_visibility,
                    identity_pooling=cfg.pass_c_identity_pooling,
                    vertex_gate_mode=cfg.vertex_gate_mode,
                    vertex_gate_delta_cap=cfg.vertex_gate_delta_cap,
                    vertex_gate_delta_beta=cfg.vertex_gate_delta_beta,
                    temporal_velocity_weight_in_pass_c=cfg.temporal_velocity_weight_in_pass_c,
                )
            else:
                raise ValueError(f"unknown pass type in pass_pattern: {pass_type!r}")

            if is_main_process(rank) and step % cfg.log_interval_steps == 0:
                extra_weights = {
                    "vertex_smooth": cfg.temporal_vertex_smoothness_weight,
                    "vel_expr": cfg.temporal_velocity_weight_in_pass_c,
                    "vel_jaw": cfg.temporal_velocity_weight_in_pass_c,
                    "vel_camera": cfg.temporal_velocity_weight_in_pass_c,
                    "vel_shape": cfg.temporal_velocity_weight_in_pass_c,
                }
                print(f"step {step} pass={pass_type}: {metrics} weighted={weighted_metrics(metrics, extra_weights)}")
                if wandb.run is not None:
                    wandb.log(flatten_metrics(metrics, prefix=f"train/pass_{pass_type}"), step=step)

            if is_main_process(rank) and (step + 1) % cfg.checkpoint_interval_steps == 0:
                path = save_checkpoint(
                    cfg.checkpoint_dir, step, {"svit": svit, "heads": heads, "unet": unet, "tt": tt}, optimizer,
                    num_expression_params=cfg.num_expression_params,
                )
                print(f"saved checkpoint: {path}")

            # Runs on EVERY rank (not is_main_process-gated): each rank evaluates
            # its own shard of the fixed dev-clip subset in parallel, then
            # aggregate_and_print_eval_results' collective gather needs every
            # rank to reach it together. Entirely inside no_grad, never touches
            # optimizer/gradients, so it cannot perturb training state.
            if cfg.eval.interval_steps > 0 and step % cfg.eval.interval_steps == 0:
                svit.eval()
                heads.eval()
                tt.eval()
                local_results = run_periodic_eval_local(
                    svit, tt, heads, flame, renderer, eval_loaders, device,
                    pool_identity=cfg.pass_c_identity_pooling,
                )
                aggregate_and_print_eval_results(local_results, rank, world_size, step)
                svit.train()
                heads.train()
                tt.train()

        # Unconditional final save: cfg.num_steps isn't guaranteed to be a
        # multiple of checkpoint_interval_steps (and either could change
        # later), so the interval check above alone could finish training
        # without ever saving the final weights. A duplicate save (if the
        # last loop iteration's interval check ALSO just fired for this same
        # step) just overwrites the same file with identical content -
        # harmless.
        if is_main_process(rank) and step >= start_step:
            path = save_checkpoint(
                cfg.checkpoint_dir, step, {"svit": svit, "heads": heads, "unet": unet, "tt": tt}, optimizer,
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
    parser = argparse.ArgumentParser(description="Stage 2 training (implementation-plan.md Sec 7).")
    parser.add_argument("--config", type=str, default="training/config/stage2.yaml")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    if cfg.wandb_run_name is None:
        cfg.wandb_run_name = Path(cfg.checkpoint_dir).name
    train(cfg, checkpoint_pth=cfg.checkpoint_pth)
    print("Training done!")


if __name__ == "__main__":
    main()
