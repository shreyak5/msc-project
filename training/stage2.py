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
meshes ignored" - no 2D/3D split here, unlike Pass A). Alternates between
updating (tokens + SViT + heads) and (UNet) every call, per the
`update_encoder` flag the caller passes in (SMIRK's alternating freeze,
preventing the UNet from compensating for encoder errors) - TT is frozen in
both alternations (never called, same reasoning as Pass A). Identity cycle
loss is applied on both alternations regardless (Sec 6: a deliberate deviation
from SMIRK, where the shape encoder is frozen throughout the whole pass).

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

import torch
from torch.nn.parallel import DistributedDataParallel

from dataset_processing.dataloading.combined_loader import build_combined_loader
from dataset_processing.dataloading.config import load_dataloader_config
from model import constants
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
from model.losses.landmark import eye_closure_loss, fan_boundary_loss, lip_closure_loss, mediapipe_landmark_loss
from model.losses.mesh import build_region_weights, region_weighted_mesh_loss
from model.losses.mica_shape import mica_shape_loss
from model.losses.photometric import VGGPerceptualLoss, photometric_loss
from model.losses.temporal_smoothness import velocity_penalty
from model.temporal import TemporalTransformer
from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import Stage2Config, load_stage2_config
from training.distributed import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from training.loss_utils import concat_category_fields, gated_loss, next_batch, regularization_loss
from training.losses_3d import compute_3d_losses

_REPO_ROOT_RELATIVE_FARL_PATH = "pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"


def _render_and_reconstruct(
    flame: FLAME, renderer: Renderer, unet: UNetGenerator, face_probabilities: torch.Tensor,
    encoded: dict[str, torch.Tensor], pixel_values: torch.Tensor, face_mask: torch.Tensor,
    valid_recon: torch.Tensor,
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
    as zeros in `reconstructed`, never read by the gated losses."""
    cam_for_proj = torch.cat([encoded["scale"], encoded["translation"]], dim=-1)

    flame_out = flame(encoded["shape"], encoded["expression"], encoded["jaw"], encoded["eyelid"], encoded["rotation"])
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

    photometric is additionally spatially masked to just the face region
    (batch_2d["face_mask"], the same XSeg mask _render_and_reconstruct inverts
    to build masking()'s background_mask) - see photometric_loss's own
    docstring for why (unmasked, the background's near-free reconstruction
    dilutes the gradient signal on the face region). VGG stays unmasked: its
    input must remain the full natural image (a partially blacked-out image
    would corrupt its pretrained features), and masking its loss instead
    would require resizing the mask independently per block - left for a
    follow-up if the face region still lacks structure after this change."""
    valid_recon = batch_2d["flag_face_mask_valid"]
    reconstructed, projected_fan, projected_mp = _render_and_reconstruct(
        flame, renderer, unet, face_probabilities, encoded, batch_2d["pixel_values"], batch_2d["face_mask"],
        valid_recon,
    )

    photometric = gated_loss(
        photometric_loss, valid_recon, reconstructed, batch_2d["pixel_values"], batch_2d["face_mask"].unsqueeze(1)
    )
    vgg = gated_loss(vgg_loss, valid_recon, reconstructed, batch_2d["pixel_values"])
    emotion_term = gated_loss(
        lambda r, t: emotion_loss(r, t, emotion_net), valid_recon, reconstructed, batch_2d["pixel_values"]
    )

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

    loss_excluding_emotion = (
        constants.PHOTOMETRIC_LOSS_WEIGHT * photometric
        + constants.VGG_LOSS_WEIGHT * vgg
        + constants.LANDMARK_LOSS_WEIGHT * landmark_loss
        + constants.CLOSURE_LOSS_WEIGHT * closure_loss
        + constants.MICA_SHAPE_LOSS_WEIGHT * mica_loss
        + reg_loss
    )
    metrics = {
        "photometric": photometric.item(), "vgg": vgg.item(), "emotion": emotion_term.item(),
        "landmark": landmark_loss.item(), "closure": closure_loss.item(),
        "mica": mica_loss.item(), "reg_2d": reg_loss.item(),
    }
    return loss_excluding_emotion, emotion_term, metrics


def compute_2d_reconstruction_losses(
    svit: SViT, heads: ComponentHeads, flame: FLAME, renderer: Renderer, unet: UNetGenerator,
    emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss, face_probabilities: torch.Tensor,
    batch_2d: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Pass A's 2D reconstruction losses: encode_image, then the shared loss
    aggregation - see _compute_2d_reconstruction_losses_from_encoded's own
    docstring for the full loss list and the (loss_excluding_emotion,
    emotion_term, metrics) return-shape rationale."""
    encoded = encode_image(svit, heads, batch_2d["pixel_values"])
    return _compute_2d_reconstruction_losses_from_encoded(
        flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, encoded, batch_2d,
    )


def run_pass_a(
    svit: SViT, heads: ComponentHeads, flame: FLAME, renderer: Renderer, unet: UNetGenerator,
    emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss, region_weights: torch.Tensor,
    face_probabilities: torch.Tensor, optimizer: torch.optim.Optimizer,
    batch_2d: dict[str, torch.Tensor], batch_3d: dict[str, torch.Tensor], subject_ids_3d: list[str], device: str,
) -> dict[str, float]:
    """One Pass A training step: computes losses, applies the two-backward-call
    emotion carve-out, and steps the optimizer - unlike Stage 1's compute_2d_losses/
    compute_3d_losses (which just return a loss for a caller-level single backward),
    this owns its own zero_grad/backward/step calls, since the emotion carve-out
    can't be expressed as a single combined backward (see module docstring).

    Explicitly sets svit/heads/unet requires_grad_(True) at entry rather than
    assuming it's already the case: Pass B alternates svit/heads and unet's
    requires_grad between calls (see run_pass_b), and requires_grad is a
    persistent property of the nn.Module, not reset between calls - without
    this, a Pass A call immediately following a Pass B call that happened to
    leave the encoder frozen would silently train nothing on it that step."""
    for p in list(svit.parameters()) + list(heads.parameters()) + list(unet.parameters()):
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
    )
    loss_2d_scaled = constants.LOSS_BALANCE_2D * loss_2d_excl_emotion
    emotion_loss_scaled = constants.LOSS_BALANCE_2D * constants.EMOTION_LOSS_WEIGHT * emotion_term

    # retain_graph=True: the UNet's output tensor (and everything upstream of
    # it - UNet, FLAME, encoder) is reused by emotion_loss_scaled's backward
    # below, so the graph can't be freed after this first call.
    loss_2d_scaled.backward(retain_graph=True)

    unet_params = list(unet.parameters())
    for p in unet_params:
        p.requires_grad_(False)
    # emotion_term is gated by flag_face_mask_valid - if EVERY sample in
    # batch_2d happens to be invalid, it's a disconnected zero tensor with no
    # grad_fn (same edge case already handled in run_pass_b), and calling
    # backward() on it would crash. loss_2d_scaled doesn't have this problem
    # (its own regularization term is ungated, always grad-connected).
    if emotion_loss_scaled.requires_grad:
        emotion_loss_scaled.backward()
    for p in unet_params:
        p.requires_grad_(True)

    # 3D branch: forward + backward only now, after the 2D branch's graph has
    # been fully backpropped (both calls above) and freed. compute_3d_losses'
    # own regularization term is likewise always ungated/grad-connected
    # (training/losses_3d.py), so no requires_grad guard is needed here either.
    loss_3d, metrics_3d = compute_3d_losses(svit, heads, flame, region_weights, batch_3d, subject_ids_3d, device)
    loss_3d_scaled = constants.LOSS_BALANCE_3D * loss_3d
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
    batch: dict[str, torch.Tensor], update_encoder: bool,
) -> dict[str, float]:
    """One Pass B training step. update_encoder: True updates tokens/SViT/heads
    and freezes UNet; False updates UNet and freezes tokens/SViT/heads - the
    caller (outer scheduler) decides which, alternating by its own iteration
    count and configured period (see module docstring). A single combined
    backward suffices here (unlike Pass A's emotion carve-out): whichever side
    is frozen this call just doesn't accumulate gradient, no split-backward
    trick needed. Explicitly sets requires_grad_ for BOTH sides every call
    (not just the one being turned on) for the same cross-pass-contamination
    reason documented in run_pass_a."""
    for p in list(svit.parameters()) + list(heads.parameters()):
        p.requires_grad_(update_encoder)
    for p in unet.parameters():
        p.requires_grad_(not update_encoder)

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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pass C's per-frame 2D-video losses - the SAME loss set as Pass A
    (_compute_2d_reconstruction_losses_from_encoded), just fed by encode_video's
    flattened output instead of encode_image's, and with real_frame_mask
    (real frame vs. tail padding) pre-ANDed into every flag_*_valid field
    before calling the shared function, so padded frames never contribute to
    any of these losses (Pass A never needed this - its batches are never
    padded)."""
    batch_size, num_frames = real_frame_mask.shape
    encoded_flat = {k: _flatten(v, batch_size, num_frames) for k, v in encoded.items()}
    batch_flat = {k: _flatten(v, batch_size, num_frames) for k, v in batch_2d_video.items()}
    real_frame_mask_flat = _flatten(real_frame_mask, batch_size, num_frames)

    for flag_key in ("flag_face_mask_valid", "flag_landmarks_fan_valid", "flag_landmarks_mp_valid", "flag_mica_valid"):
        batch_flat[flag_key] = real_frame_mask_flat & batch_flat[flag_key]

    loss_excluding_emotion, emotion_term, metrics = _compute_2d_reconstruction_losses_from_encoded(
        flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, encoded_flat, batch_flat,
    )
    total = loss_excluding_emotion + constants.EMOTION_LOSS_WEIGHT * emotion_term
    return total, metrics


def compute_3d_video_losses(
    flame: FLAME, region_weights: torch.Tensor, encoded: dict[str, torch.Tensor],
    batch_3d_video: dict[str, torch.Tensor], real_frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pass C's per-frame 3D-video loss: mesh + regularization, no Lvc -
    clip-mode video batches use a plain sampler, not IdentityAwareBatchSampler,
    so there's no guarantee any two clips in a batch share a subject
    (identity_batch_sampler.py's own docstring already establishes clip mode
    doesn't fit the identity-pairing scheme). Regularization here is
    unconditional (not gated by real_frame_mask), matching compute_3d_losses'
    own Pass A/Stage 1 pattern (never gated by anything there either) - a
    padded frame's contribution just duplicates the real last frame's own
    regularization value, the same minor, accepted imprecision as elsewhere,
    not worth an extra filtering step for."""
    batch_size, num_frames = real_frame_mask.shape
    encoded_flat = {k: _flatten(v, batch_size, num_frames) for k, v in encoded.items()}
    flame_vertices_flat = _flatten(batch_3d_video["flame_vertices"], batch_size, num_frames)
    real_frame_mask_flat = _flatten(real_frame_mask, batch_size, num_frames)

    flame_out = flame(
        encoded_flat["shape"], encoded_flat["expression"], encoded_flat["jaw"],
        encoded_flat["eyelid"], encoded_flat["rotation"],
    )
    mesh_loss = gated_loss(
        lambda pred, gt: region_weighted_mesh_loss(pred, gt, region_weights),
        real_frame_mask_flat, flame_out["vertices"], flame_vertices_flat,
    )
    reg_loss = regularization_loss(encoded_flat)

    total = constants.MESH_LOSS_LAMBDA * mesh_loss + reg_loss
    metrics = {"mesh": mesh_loss.item(), "reg_3d": reg_loss.item()}
    return total, metrics


def compute_temporal_smoothness_losses(
    encoded: dict[str, torch.Tensor], real_frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sec 6: velocity penalty (L1, discourages frame-to-frame jumps) applied
    uniformly to expression+eyelid, jaw, camera scale+rotation, and shape.
    Computed on encode_video's DECODED per-frame params directly ((B,N,...),
    before any flattening - needs the time axis to diff across), using the
    valid_mask-aware velocity_penalty (model/losses/temporal_smoothness.py) so
    a difference spanning into tail-padding never contaminates the loss."""
    expr_eyelid = torch.cat([encoded["expression"], encoded["eyelid"]], dim=-1)
    camera_rotation = torch.cat([encoded["scale"], encoded["rotation"]], dim=-1)

    vel_expr = velocity_penalty(expr_eyelid, real_frame_mask)
    vel_jaw = velocity_penalty(encoded["jaw"], real_frame_mask)
    vel_camera = velocity_penalty(camera_rotation, real_frame_mask)
    vel_shape = velocity_penalty(encoded["shape"], real_frame_mask)

    total = constants.TEMPORAL_VELOCITY_WEIGHT * (vel_expr + vel_jaw + vel_camera + vel_shape)
    metrics = {
        "vel_expr": vel_expr.item(), "vel_jaw": vel_jaw.item(),
        "vel_camera": vel_camera.item(), "vel_shape": vel_shape.item(),
    }
    return total, metrics


def run_pass_c(
    svit: SViT, tt: TemporalTransformer, heads: ComponentHeads, flame: FLAME, renderer: Renderer,
    unet: UNetGenerator, emotion_net: EmotionNet, vgg_loss: VGGPerceptualLoss, region_weights: torch.Tensor,
    face_probabilities: torch.Tensor, optimizer: torch.optim.Optimizer,
    batch_2d_video: dict[str, torch.Tensor], batch_3d_video: dict[str, torch.Tensor],
) -> dict[str, float]:
    """One Pass C training step. Only TT updates - svit/heads/unet are
    explicitly frozen at entry (same defensive requires_grad_ pattern as Pass
    A/B), but still run forward (see module docstring: frozen doesn't mean
    skipped, gradient still flows through them to reach TT).

    2d_video and 3d_video get their OWN separate encode_video calls (unlike an
    earlier version of this function, which concatenated them into one) -
    required to actually reduce peak memory, not just reorder compute:
    retain_graph=True is all-or-nothing (keeps the ENTIRE graph reachable from
    whatever backward() was called on, not just "the shared part"), so a
    single combined encode_video call would force loss_2d's own heavy
    Renderer/UNet/VGG/EmotionNet graph to stay resident for as long as
    loss_3d/loss_temporal still need the shared encoded tensor to backward
    through - defeating any memory benefit (confirmed via a real GPU OOM in
    Pass A's structurally similar case; see run_pass_a's own docstring). Two
    separate (redundant but comparatively cheap) SViT+TT+heads forward passes
    buys a real reduction: the two categories' HEAVY reconstruction graphs are
    never resident simultaneously.

    Backward order matters for correctness here, not just memory: loss_temporal
    is computed by concatenating encoded_2d+encoded_3d back together - exactly
    reproducing the original single pooled masked-mean velocity_penalty
    aggregation across BOTH categories combined (model/losses/
    temporal_smoothness.py's own denominator is a single pooled valid-term
    count) - splitting that computation per-category and adding the two means
    back together would NOT be equivalent, since each category can contribute
    a different number of valid terms. loss_temporal is backwarded FIRST, with
    retain_graph=True - cheap to retain at this point, since neither
    category's heavy reconstruction graph has been built yet. Only THEN are
    loss_2d and loss_3d each built and backwarded (no retain_graph needed for
    either - nothing needs their specific graphs again afterward), one at a
    time, so at no point are both heavy per-category graphs resident
    together."""
    for p in list(svit.parameters()) + list(heads.parameters()) + list(unet.parameters()):
        p.requires_grad_(False)
    for p in tt.parameters():
        p.requires_grad_(True)

    num_frames = batch_2d_video["pixel_values"].shape[1]
    device = batch_2d_video["pixel_values"].device
    assert batch_3d_video["pixel_values"].shape[1] == num_frames, (
        "2d_video and 3d_video categories must share the same clip length (dataloader.yaml's max_frames)"
    )

    def _encode_category(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["pixel_values"].shape[0]
        frame_indices = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1)
        return encode_video(
            svit, tt, heads, batch["pixel_values"], batch["visibility_ratio"],
            frame_indices, batch["flag_visibility_valid"], batch["valid_mask"],
        )

    encoded_2d = _encode_category(batch_2d_video)
    encoded_3d = _encode_category(batch_3d_video)
    real_frame_mask_2d = batch_2d_video["valid_mask"]
    real_frame_mask_3d = batch_3d_video["valid_mask"]

    optimizer.zero_grad()

    # Temporal smoothness first: needs both categories combined to reproduce
    # the exact original pooled aggregation (see docstring above).
    encoded_combined = {k: torch.cat([encoded_2d[k], encoded_3d[k]], dim=0) for k in encoded_2d}
    real_frame_mask_combined = torch.cat([real_frame_mask_2d, real_frame_mask_3d], dim=0)
    loss_temporal, metrics_temporal = compute_temporal_smoothness_losses(encoded_combined, real_frame_mask_combined)
    # retain_graph=True: encoded_2d/encoded_3d's own graphs are still needed
    # by loss_2d/loss_3d's backwards below - cheap to retain here, since
    # neither category's heavy reconstruction graph has been built yet.
    loss_temporal.backward(retain_graph=True)

    # Regularization is NOT computed separately here: compute_2d_video_losses
    # already includes its own (via the shared _compute_2d_reconstruction_
    # losses_from_encoded, same as Pass A) and compute_3d_video_losses now
    # does too - each category's own function owns its own regularization
    # term, matching Pass A's per-category pattern, rather than a combined
    # one here that would double-count the 2D-video portion's contribution.
    loss_2d, metrics_2d = compute_2d_video_losses(
        flame, renderer, unet, emotion_net, vgg_loss, face_probabilities, encoded_2d, batch_2d_video, real_frame_mask_2d,
    )
    loss_2d.backward()

    loss_3d, metrics_3d = compute_3d_video_losses(flame, region_weights, encoded_3d, batch_3d_video, real_frame_mask_3d)
    loss_3d.backward()

    optimizer.step()

    total_loss = loss_2d.item() + loss_3d.item() + loss_temporal.item()
    metrics = {
        "total": total_loss, "2d": metrics_2d, "3d": metrics_3d, "temporal": metrics_temporal,
    }
    return metrics


def train(cfg: Stage2Config, checkpoint_pth: str | None = None) -> None:
    """The outer three-pass scheduler: round-robins through cfg.pass_pattern
    (default ["A","B","C"]), drawing batches from one of two independently-
    cycling loaders (frame_pool for Pass A/B, clip for Pass C - see module
    docstring) via training.loss_utils.next_batch, which restarts each loader
    on exhaustion rather than stopping (there's no single "epoch" spanning
    both loaders, since they're drawn from at different relative rates under
    round-robin - see training/config.py's Stage2Config docstring)."""
    torch.manual_seed(cfg.seed)
    rank, world_size, local_rank, device = setup_distributed(fallback_device=cfg.device)

    try:
        svit = SViT().to(device)
        load_farl_pretrained(svit, _REPO_ROOT_RELATIVE_FARL_PATH)
        heads = ComponentHeads().to(device)
        flame = FLAME().to(device)
        renderer = Renderer(flame.faces_tensor).to(device)
        unet = UNetGenerator().to(device)
        tt = TemporalTransformer().to(device)
        emotion_net = EmotionNet().to(device)
        vgg_loss = VGGPerceptualLoss().to(device)
        region_weights = build_region_weights().to(device)
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
            loaded_step = load_checkpoint(
                checkpoint_pth, {"svit": svit, "heads": heads, "unet": unet, "tt": tt}, optimizer, device,
            )
            start_step = loaded_step + 1
        else:
            # One-time seed from Stage 1 - svit+heads only, no optimizer, no
            # unet/tt (Stage 1 never saved any) - only on a fresh run (no
            # Stage-2-own checkpoint to resume from). See Stage2Config's own
            # docstring for why this is skipped on a resume.
            load_checkpoint(cfg.stage1_checkpoint_pth, {"svit": svit, "heads": heads})

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
        keys_3d_video = ["pixel_values", "flame_vertices", "visibility_ratio", "flag_visibility_valid", "valid_mask"]

        pass_b_call_count = 0
        step = start_step - 1  # in case cfg.num_steps <= start_step, so the final-checkpoint save below still has a defined step

        for step in range(start_step, cfg.num_steps):
            pass_type = cfg.pass_pattern[step % len(cfg.pass_pattern)]

            if pass_type == "A":
                batch, frame_pool_iterator, frame_pool_epoch = next_batch(frame_pool_loader, frame_pool_iterator, frame_pool_epoch)
                batch_2d = concat_category_fields(batch, ["2d_image", "2d_video"], keys_2d_a, device)
                batch_3d = concat_category_fields(batch, ["3d_image", "3d_video"], keys_3d_a, device)
                subject_ids_3d = batch["3d_image"]["subject_id"] + batch["3d_video"]["subject_id"]
                metrics = run_pass_a(
                    svit, heads, flame, renderer, unet, emotion_net, vgg_loss, region_weights,
                    face_probabilities, optimizer, batch_2d, batch_3d, subject_ids_3d, device,
                )
            elif pass_type == "B":
                batch, frame_pool_iterator, frame_pool_epoch = next_batch(frame_pool_loader, frame_pool_iterator, frame_pool_epoch)
                batch_b = concat_category_fields(batch, ["2d_image", "2d_video", "3d_image", "3d_video"], keys_b, device)
                update_encoder = (pass_b_call_count // cfg.pass_b_alternation_period) % 2 == 0
                metrics = run_pass_b(
                    svit, heads, flame, renderer, unet, face_probabilities, templates, optimizer, batch_b, update_encoder,
                )
                pass_b_call_count += 1
            elif pass_type == "C":
                batch, clip_iterator, clip_epoch = next_batch(clip_loader, clip_iterator, clip_epoch)
                batch_2d_video = {k: batch["2d_video"][k].to(device) for k in keys_2d_video}
                batch_3d_video = {k: batch["3d_video"][k].to(device) for k in keys_3d_video}
                metrics = run_pass_c(
                    svit, tt, heads, flame, renderer, unet, emotion_net, vgg_loss, region_weights,
                    face_probabilities, optimizer, batch_2d_video, batch_3d_video,
                )
            else:
                raise ValueError(f"unknown pass type in pass_pattern: {pass_type!r}")

            if is_main_process(rank) and step % cfg.log_interval_steps == 0:
                print(f"step {step} pass={pass_type}: {metrics}")

            if is_main_process(rank) and (step + 1) % cfg.checkpoint_interval_steps == 0:
                path = save_checkpoint(
                    cfg.checkpoint_dir, step, {"svit": svit, "heads": heads, "unet": unet, "tt": tt}, optimizer,
                )
                print(f"saved checkpoint: {path}")

        # Unconditional final save: cfg.num_steps isn't guaranteed to be a
        # multiple of checkpoint_interval_steps (and either could change
        # later), so the interval check above alone could finish training
        # without ever saving the final weights. A duplicate save (if the
        # last loop iteration's interval check ALSO just fired for this same
        # step) just overwrites the same file with identical content -
        # harmless.
        if is_main_process(rank) and step >= start_step:
            path = save_checkpoint(cfg.checkpoint_dir, step, {"svit": svit, "heads": heads, "unet": unet, "tt": tt}, optimizer)
            print(f"saved final checkpoint: {path}")

        if is_main_process(rank):
            print("DONE!")
    finally:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 training (implementation-plan.md Sec 7).")
    parser.add_argument("--config", type=str, default="training/config/stage2.yaml")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    train(cfg, checkpoint_pth=cfg.checkpoint_pth)
    print("Training done!")


if __name__ == "__main__":
    main()
