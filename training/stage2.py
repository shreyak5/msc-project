"""Stage 2 training loop (implementation-plan.md Sec 7, "Stage 2 - SMIRK
training + temporal training"): three alternating passes (A reconstruction,
B augmentation/cycle, C temporal) that bring in the UNet, TT, and the
remaining losses Stage 1 never touched.

This module implements all three passes plus the outer three-pass scheduler
(train()/main()) that round-robins between them and manages the two loaders,
checkpointing, and DDP wiring.

Pass A (Sec 7): 2D batches go through the full reconstruction path (mask ->
sample 1% pixels -> UNet -> photometric/VGG/landmark/emotion/MICA/regularization
losses); 3D batches get mesh + Lvc, identical to Stage 1 - reuses
training.losses_3d.compute_3d_losses as-is rather than duplicating it. Updates
tokens/SViT/heads/UNet; TT is frozen (simply never called in this pass, since
Pass A operates on individual frames/images, not clips). UNet is additionally
frozen with respect to the emotion loss only (Sec 6) - since a single combined
backward() can't selectively exclude one loss term's gradient from one
component while other terms in the same pass still update it, this is done as
two separate backward() calls: the main loss (retain_graph=True, since the
UNet's output tensor is reused by the second call), then the emotion term
alone with the UNet's own parameters' requires_grad temporarily set to False.

Pass B (Sec 7): augments the encoded expression/jaw/eyelid, re-renders +
transfers real pixels to the augmented mesh's new projected locations, runs
the UNet, then re-encodes the result and checks whether the encoder recovers
the augmented target (expression cycle loss) and the original identity
(identity cycle loss). Operates on ALL FOUR categories' images combined into
one undifferentiated batch (Sec 7: "3D datasets contribute their 2D images,
meshes ignored" - no 2D/3D split here, unlike Pass A). Cycles through three
modes across consecutive Pass B calls - "encoder" ((tokens + SViT + heads)
update, UNet frozen), "unet" (UNet updates, (tokens + SViT + heads) frozen),
and "joint" (both update together) - per the `mode` string the caller passes
in, itself derived from the outer scheduler's own call count and the
configured (pass_b_encoder_steps, pass_b_unet_steps, pass_b_joint_steps) cycle
(training/config.py). Pure "encoder"/"unet" alternation (the original
SMIRK-style freeze, preventing the UNet from compensating for encoder errors)
is the config default; "joint" is an added third mode, not part of SMIRK's
original scheme. TT is frozen in every mode (never called, same reasoning as
Pass A). Identity cycle loss is applied in every mode regardless (Sec 6: a
deliberate deviation from SMIRK, where the shape encoder is frozen throughout
the whole pass).

Pass C (Sec 7): full SViT -> TT -> ComponentHeads pipeline on clips
(model.encoding.encode_video) - SViT tokens can't be precomputed since SViT is
still training in passes A/B. Only TT updates; SViT/heads/UNet are frozen
(explicitly, at entry, same defensive requires_grad_ pattern as Pass A/B) but
still run forward (frozen doesn't mean skipped - gradient still needs to flow
back through their forward computation to reach TT, which is upstream of them
in the graph, the same "frozen but not detached" pattern Pass A's emotion
carve-out already relies on). A single combined backward suffices (unlike Pass
A): every loss term in this pass shares the same single "only TT updates"
gradient path, no differential freezing needed within the pass itself.

2D video gets the SAME full loss set as Pass A's 2D reconstruction path
(photometric/VGG/landmark+closure/MICA/emotion/regularization), via the exact
same _compute_2d_reconstruction_losses_from_encoded (fed by encode_video's
flattened (B*N,...) output instead of encode_image's (B,...), not a second
driftable copy of the loss list) - a deliberate extension beyond Sec 7's
literal (narrower) text, since Pass C's UNet is unconditionally frozen for
every loss term here (unlike Pass A, where the emotion carve-out exists
specifically because UNet is trainable for every OTHER term), so there's no
architectural reason to exclude MICA/emotion here. 3D video gets mesh loss
only, no Lvc (clip-mode video was already established elsewhere -
identity_batch_sampler.py's own docstring - as not fitting the identity-
pairing scheme). Both are additionally gated by real_frame_mask (excluding
tail-padding, which Pass A never needed since its batches are never padded).

Temporal smoothness is computed on encode_video's DECODED per-frame FLAME/
camera params (after ComponentHeads, not on raw pre-head tokens) - directly on
the (B,N,...) shape, before any flattening, since it needs the time axis:
a velocity penalty, applied uniformly to expression+eyelid, jaw, camera
scale+rotation, and shape - using the valid_mask-aware velocity_penalty
(model/losses/temporal_smoothness.py) so a difference spanning into
tail-padding never contaminates the loss.
"""

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
    """Shared mask -> sample 1% pixels -> UNet reconstruction path (Sec 7 Pass
    A/C): given already-encoded FLAME params (from encode_image or, after
    flattening, encode_video), renders the mesh, samples sparse real pixels,
    composites the masked input, and runs the UNet. Returns (reconstructed,
    projected_fan, projected_mp). Pass A's compute_2d_reconstruction_losses and
    Pass C's compute_2d_video_losses both call this rather than duplicating it -
    they differ in which losses they compute from `reconstructed`, not in how
    it's produced.

    valid_recon: (B,) bool - batch_2d["flag_face_mask_valid"]. flame/renderer
    still run over the FULL batch (projected_fan/projected_mp are needed
    regardless - landmark losses are gated by their own, separately-diverging
    validity flags, not this one). But mesh_based_mask_uniform_faces/masking/
    unet only run on valid_recon rows: an invalid row's encoded FLAME/camera
    params are never supervised (no detected face -> no photometric/VGG/
    emotion gradient reaches them), so they can come out numerically extreme
    and crash mesh_based_mask_uniform_faces's torch.multinomial call - and
    their `reconstructed` output is unused anyway, since photometric/VGG/
    emotion losses already gate on this same flag. Non-valid rows come back
    as zeros in `reconstructed`, never read by the gated losses.

    precomputed_flame_out: occlusion-experiment1.md Change 2's FLAME-forward-once
    optimization (run_pass_c) - when given (a dict with the same "vertices"/
    "landmarks_fan"/"landmarks_mp" keys FLAME.forward() returns, already computed
    on THIS SAME `encoded`, in the same row order), skips this function's own
    flame(...) call and uses it directly - mirrors dataset_processing/dataloading/
    face_parsing_cache.py's precomputed_crop parameter on compute_face_parsing
    (skip recomputation when the caller already has the result). None (default,
    every Pass A/B call site) reproduces the original always-call-flame-here
    behavior exactly."""
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
    """The full 2D reconstruction loss set (Sec 6: photometric + VGG + landmark
    (+closure) + MICA + emotion + regularization), given ALREADY-ENCODED FLAME
    params - factored out of compute_2d_reconstruction_losses so Pass C's
    compute_2d_video_losses can reuse the exact same loss-aggregation code
    (fed by encode_video's flattened output instead of encode_image's), rather
    than a second, driftable copy of this whole loss list. Any padding-
    exclusion Pass C needs is the caller's job (pre-AND real_frame_mask into
    each flag_*_valid entry of `batch_2d` before calling this) - this function
    itself knows nothing about clip padding, only about validity flags.

    Returns (loss_excluding_emotion, emotion_term, metrics) - loss_excluding_
    emotion and emotion_term are kept as separate tensors (rather than summed
    into one scalar) so run_pass_a can apply the UNet-frozen-for-emotion-only
    backward carve-out described in this module's docstring (Pass C doesn't
    need this split - UNet is unconditionally frozen there regardless of loss
    term, so its caller just adds emotion back in before a single backward);
    metrics still reports emotion alongside the other 2D-batch losses (Sec 6
    lists it as a 2D-batch loss), it's just not folded into the returned loss
    tensor.

    All of photometric/VGG/emotion are gated by flag_face_mask_valid (same
    gated_loss mechanism landmark/MICA already use): a sample with no detected
    face has an all-zero face_mask fallback, which would make masking()'s output
    degenerate - training the reconstruction path against that would be
    training against garbage, not a real supervision signal, so those rows are
    excluded the same way an invalid landmark/MICA target already would be.

    photometric is currently unmasked (full-image L1) - photometric_loss
    supports an optional face-region mask (see its own docstring), tried
    briefly to concentrate gradient on the harder face region rather than the
    near-free background, but reverted alongside the VGG_LOSS_WEIGHT cut: the
    combination left the UNet's reconstruction a noisy/checkerboard mess
    rather than the smoother (if blurry) output it produced unmasked - the
    mask param is kept, not deleted, in case it's worth revisiting once VGG's
    weight is back to providing enough perceptual/structural regularization
    on its own.

    occlusion_mask/occlusion_loss_weight: Pass C's synthetic-occlusion
    upweighting only (compute_2d_video_losses) - Pass A's own call site
    (compute_2d_reconstruction_losses) never passes these, so its behavior is
    unchanged. When occlusion_loss_weight != 1.0, the same gated-loss
    aggregation is evaluated a SECOND time, restricted to occlusion_mask's
    rows, and added in at (occlusion_loss_weight - 1.0)x on top of the normal
    1x pass every row already gets - reusing `reconstructed`/`projected_fan`/
    `projected_mp` from the single _render_and_reconstruct call above (FLAME/
    renderer already run over the full batch regardless of valid_recon, so
    this needs no second render). reg_loss is deliberately excluded from this
    upweighting - it's a global param regularizer with no per-frame validity
    concept, not a per-frame supervision signal occlusion-weighting applies to.

    landmark_occlusion_masking: occlusion-experiment1.md Change 1
    (Stage2Config.landmark_occlusion_masking), extended by occlusion-
    experiment1-passA.md to also cover Pass A's own call site
    (compute_2d_reconstruction_losses), not just Pass C's - both callers read
    the same config field. False (default) reproduces the exact original
    unmasked landmark/closure loss. When True, per-landmark occlusion masks
    (model/losses/landmark.py's landmark_visibility_mask, keyed off
    batch_2d["face_mask"] + the GT/target landmarks, NOT the
    projected/predicted ones) are computed once here and fed as each loss
    function's new `mask` argument.

    precomputed_flame_out: forwarded to _render_and_reconstruct - see its own
    docstring."""
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
    """Pass A's 2D reconstruction losses: encode_image, then the shared loss
    aggregation - see _compute_2d_reconstruction_losses_from_encoded's own
    docstring for the full loss list and the (loss_excluding_emotion,
    emotion_term, metrics) return-shape rationale.

    landmark_occlusion_masking: forwarded straight to
    _compute_2d_reconstruction_losses_from_encoded - see its own docstring
    (occlusion-experiment1-passA.md extended this from Pass-C-only to also
    cover Pass A's landmark/closure loss)."""
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
    """One Pass A training step: computes losses, applies the two-backward-call
    emotion carve-out, and steps the optimizer - unlike Stage 1's compute_2d_losses/
    compute_3d_losses (which just return a loss for a caller-level single backward),
    this owns its own zero_grad/backward/step calls, since the emotion carve-out
    can't be expressed as a single combined backward (see module docstring).

    Explicitly sets svit/heads/unet requires_grad_ at entry rather than assuming
    it's already the case: Pass B alternates svit/heads and unet's requires_grad
    between calls (see run_pass_b), and requires_grad is a persistent property of
    the nn.Module, not reset between calls - without this, a Pass A call
    immediately following a Pass B call that happened to leave the encoder frozen
    would silently train nothing on it that step.

    freeze_encoder (Stage2Config.freeze_encoder): keeps svit/heads at
    requires_grad_(False) even here - a UNet-warmup phase (train the
    reconstruction pathway against a STABLE, Stage-1-converged geometry signal
    before letting the encoder move again), not just a per-pass toggle.
    landmark/mesh/mica/reg still get computed and logged as usual for
    monitoring (cheap relative to the rest of the forward pass), but with
    svit/heads frozen their backward produces no gradient anywhere - only
    photometric/VGG/emotion (which route through unet) actually train
    anything during this phase. unet itself is never frozen by this flag.

    landmark_occlusion_masking: forwarded straight to
    compute_2d_reconstruction_losses (Stage2Config.landmark_occlusion_masking) -
    occlusion-experiment1-passA.md's extension of Change 1 to Pass A."""
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
    """Pass B's augmentation/cycle loss. batch combines ALL FOUR categories'
    pixel_values/face_mask/flag_face_mask_valid into one undifferentiated image
    batch (see module docstring) - there's no 2D/3D split here, unlike Pass A.

    The render -> sample -> mask -> UNet chain below only runs on
    flag_face_mask_valid rows (valid_recon), same reasoning as
    _render_and_reconstruct: an invalid row's encoded FLAME/camera params are
    never supervised (no detected face), so they can come out numerically
    extreme and crash mesh_based_mask_uniform_faces's torch.multinomial call -
    and unlike Pass A/C, nothing else in this pass needs an invalid row's
    render output (no separate landmark loss here), so it's sliced out
    upstream of FLAME/the renderer entirely rather than only at the very end."""
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
    """One Pass B training step. mode: "encoder" updates tokens/SViT/heads and
    freezes UNet; "unet" updates UNet and freezes tokens/SViT/heads; "joint"
    updates both together - the caller (outer scheduler) decides which,
    cycling by its own Pass B call count and the configured
    (pass_b_encoder_steps, pass_b_unet_steps, pass_b_joint_steps) pattern (see
    module docstring). A single combined backward suffices here (unlike Pass
    A's emotion carve-out): whichever side is frozen this call just doesn't
    accumulate gradient, no split-backward trick needed - and "joint" needs no
    special-casing either, since a combined backward over two fully-unfrozen
    components is the ordinary case. Explicitly sets requires_grad_ for BOTH
    sides every call (not just the one(s) being turned on) for the same
    cross-pass-contamination reason documented in run_pass_a."""
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
    """Pass C's per-frame 2D-video losses - the SAME loss set as Pass A
    (_compute_2d_reconstruction_losses_from_encoded), just fed by encode_video's
    flattened output instead of encode_image's, and with real_frame_mask
    (real frame vs. tail padding) pre-ANDed into every flag_*_valid field
    before calling the shared function, so padded frames never contribute to
    any of these losses (Pass A never needed this - its batches are never
    padded).

    occlusion_mask: (B, N) bool, optional - run_pass_c's synthetic-occlusion
    mask (see model/encoding.py's _fill_missing_frame_tokens and
    training/config.py's pass_c_synthetic_occlusion_enabled). Flattened the
    same way as every other per-frame field, then handed to
    _compute_2d_reconstruction_losses_from_encoded to upweight those frames'
    contribution to the loss by occlusion_loss_weight.

    landmark_occlusion_masking/precomputed_flame_out: forwarded straight to
    _compute_2d_reconstruction_losses_from_encoded (occlusion-experiment1.md's
    Change 1 / the FLAME-once optimization). precomputed_flame_out must already
    be flattened to (B*N, ...) - the SAME flatten this function does internally
    to encoded/batch_2d_video below (via _flatten, a pure deterministic
    reshape) - so row i of precomputed_flame_out lines up with row i of this
    function's own encoded_flat/batch_flat as long as the caller built it from
    the identical (B, N, ...) `encoded` this function also received. run_pass_c
    is the only caller that passes this; guaranteed by construction there (see
    its own docstring)."""
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
    """Sec 6: velocity penalty (L2, discourages frame-to-frame jumps) applied
    uniformly to expression+eyelid, jaw, camera scale+rotation, and shape.
    Computed on encode_video's DECODED per-frame params directly ((B,N,...),
    before any flattening - needs the time axis to diff across), using the
    valid_mask-aware velocity_penalty (model/losses/temporal_smoothness.py) so
    a difference spanning into tail-padding never contaminates the loss.

    temporal_velocity_weight: Stage2Config.temporal_velocity_weight_in_pass_c -
    overrides constants.TEMPORAL_VELOCITY_WEIGHT for this call only, so a Pass C
    experiment YAML can tune it without changing the shared module-level
    constant (which every other TEMPORAL_VELOCITY_WEIGHT reference elsewhere
    would otherwise also pick up). Defaults to the constant itself, reproducing
    the exact original behavior for every caller that doesn't override it.

    occlusion_mask: (B, N) bool, optional - run_pass_c's synthetic-occlusion
    mask. Converted to a (B, N) float frame_weight (occlusion_loss_weight at
    occluded positions, 1.0 elsewhere) and passed to velocity_penalty, which
    upweights any velocity term touching an occluded frame - exactly the
    frame-to-frame transition TT had to bridge using temporal context alone.

    param_smoothness_enabled: Stage2Config.param_smoothness_in_pass_c - gates the
    EXISTING param-space velocity term (expr/jaw/camera/shape) on/off; True
    (default) reproduces the original always-on behavior for every caller that
    doesn't pass this - occlusion-experiment1.md's Change 2 spec: "keep it
    implemented ..., but disable it in Pass C for this experiment (config-flag
    it off, don't delete)".

    vertices/base_region_weights/gated_region_mask/gate/
    expressive_region_smooth_weight/temporal_vertex_smoothness_weight: Change 2's
    new vertex-space, region-weighted, visibility-gated temporal smoothness term
    (model/losses/temporal_smoothness.py's vertex_velocity_penalty). `vertices`:
    (B, N, V, 3), required whenever temporal_vertex_smoothness_weight > 0 (raises
    AssertionError otherwise, along with base_region_weights/gated_region_mask/
    gate). temporal_vertex_smoothness_weight=0.0 (default) skips this term
    entirely - reproduces the exact old behavior (no vertex-space term at all)
    for every existing caller."""
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
    """One Pass C training step. Only TT updates - svit/heads/unet are
    explicitly frozen at entry (same defensive requires_grad_ pattern as Pass
    A/B), but still run forward (see module docstring: frozen doesn't mean
    skipped, gradient still flows through them to reach TT).

    2d_video only - there is no 3d_video category to encode/loss against
    (CoMA/VOCASET moved to 3d_image, since their visibility scores are
    unreliable for TT's windowed attention; see dataset_processing/
    dataloading/datasets.py and the indexers' own docstrings), so this
    simplifies to the ordinary single-category case: one encode_video call,
    temporal-smoothness loss computed directly on its output (no
    cross-category concatenation needed - an earlier version of this function
    combined a 2d_video and a 3d_video encode_video call specifically to keep
    their heavy reconstruction graphs from being resident simultaneously,
    which no longer applies with only one category).

    synthetic_occlusion_enabled/_prob/occlusion_loss_weight: training/
    config.py's pass_c_synthetic_occlusion_* fields. When enabled, some real,
    currently-visible frames (valid_mask & flag_visibility_valid) are chosen
    independently at random (Bernoulli(synthetic_occlusion_prob) per eligible
    frame) and pretended-occluded for encode_video's input only: their
    flag_visibility_valid is flipped to False (routing them through
    encode_video's existing _fill_missing_frame_tokens neighbor-averaging,
    model/encoding.py) and their visibility_ratio fed to TT is zeroed.
    batch_2d_video's own tensors are never mutated - the real, uncorrupted
    targets/flags are what the loss calls below still see, so supervision is
    against genuine ground truth. The resulting occlusion_mask is passed to
    both loss functions to upweight those frames by occlusion_loss_weight.

    landmark_occlusion_masking/base_region_weights/gated_region_mask/
    expressive_region_smooth_weight/temporal_vertex_smoothness_weight/
    param_smoothness_in_pass_c/mouth_gate_use_region_visibility:
    occlusion-experiment1.md's Change 1/Change 2. base_region_weights/
    gated_region_mask are training/stage2.py::train()'s once-built
    model/losses/mesh.py::build_gated_expressive_region_weights() output.

    identity_pooling: Stage2Config.pass_c_identity_pooling - forwarded straight
    to encode_video's own pool_identity param (model/encoding.py). False
    (default) reproduces the original per-frame shape behavior exactly. See
    stage2-config-reference.md for the full rationale.

    vertex_gate_mode/vertex_gate_delta_cap/vertex_gate_delta_beta:
    Stage2Config fields of the same name, forwarded straight to
    model/losses/temporal_smoothness.py's compute_vertex_gate (see its own
    docstring for the two modes' formulas). vertex_gate_mode="min_vis"
    (default) reproduces the original inline gate formula exactly; cap/beta
    only affect "delta_vis" mode.

    temporal_velocity_weight_in_pass_c: Stage2Config field of the same name -
    forwarded straight to compute_temporal_smoothness_losses' own
    temporal_velocity_weight param, overriding constants.TEMPORAL_VELOCITY_WEIGHT
    for the param-space velocity term (vel_expr/vel_jaw/vel_camera/vel_shape)
    in this Pass C call only. Defaults to the constant itself, reproducing the
    exact original behavior for every existing YAML.

    RESTRUCTURED FLAME flow: FLAME's forward (cheap - LBS + landmark
    regression, no UNet/VGG/renderer rasterization) now runs exactly ONCE, on
    the flattened encode_video output, BEFORE either backward call -
    previously it ran once, but only inside _render_and_reconstruct, i.e.
    after the temporal-smoothness backward had already completed. It's needed
    earlier now because Change 2's vertex-space term needs real FLAME
    vertices. The resulting flame_out is reused (not recomputed) by
    compute_2d_video_losses's own reconstruction path via the
    precomputed_flame_out parameter - FLAME never runs twice. retain_graph=
    True on the temporal-loss backward (unchanged) now also keeps flame_out's
    own (cheap) graph alive for compute_2d_video_losses' later backward - this
    doesn't reintroduce the heavy-graph-residency problem the original
    two-backward split was avoiding (that was about UNet/VGG activations, not
    FLAME's own lightweight forward)."""
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
    """Runs THIS RANK'S OWN SHARD of the fixed dev-clip subset (training/
    eval_loaders.py's build_eval_loaders shards it across every rank via
    DistributedSampler) through the same encode_video -> flame -> renderer
    forward path run_pass_c's _encode_category/_render_and_reconstruct use -
    but stops at the projected landmarks/vertices, skipping the UNet/masking/
    photometric portion entirely (not needed for landmark/temporal-smoothness
    scoring). flame_out["landmarks_fan"] is already FLAME's full 68-point set
    (the [:17] boundary-only slicing only happens in the training LOSS
    functions, not in FLAME/the renderer), so it compares directly against the
    landmarks_fan_full cache field with no extra projection work.

    pool_identity: forwarded straight to encode_video's own pool_identity param
    - should match whatever run_pass_c was called with (Stage2Config.
    pass_c_identity_pooling) so this eval's temporal_smoothness metric reflects
    the same behavior training is actually optimizing. False (default)
    reproduces the original per-frame shape behavior exactly.

    Returns RAW, unsummarized per-frame(-pair) error arrays per dataset -
    aggregation across ranks happens separately in
    aggregate_and_print_eval_results, since combining already-summarized
    per-rank statistics (e.g. averaging medians) isn't valid."""
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
    """Combines every rank's raw per-frame error arrays (run_periodic_eval_local)
    into one pooled evaluation/metrics.py summarize() per dataset/metric,
    printed only on rank 0. A single dist.gather_object collective - every rank
    reaches this at the same `step` (the training loop is lockstep-synchronized
    across ranks: `step` is the same loop variable on every rank), so this is a
    bounded, ordinary collective, not a stall.

    Combining RAW arrays (not each rank's own already-summarized mean/median/
    std) is required for correctness: those can't be recombined into the
    correct pooled statistic after the fact, especially the median."""
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
    """The outer three-pass scheduler: round-robins through cfg.pass_pattern
    (default ["A","B","C"]), drawing batches from one of two independently-
    cycling loaders (frame_pool for Pass A/B, clip for Pass C - see module
    docstring) via training.loss_utils.next_batch, which restarts each loader
    on exhaustion rather than stopping (there's no single "epoch" spanning
    both loaders, since they're drawn from at different relative rates under
    round-robin - see training/config.py's Stage2Config docstring).

    Also ramps the optimizer's LR linearly over cfg.warmup_steps at the top
    of the loop (see Stage2Config.warmup_steps' own docstring for why) -
    computed fresh from `step` every iteration rather than a stateful
    scheduler, so it's automatically correct across resumes with no extra
    checkpoint bookkeeping."""
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
