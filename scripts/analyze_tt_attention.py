"""Diagnostic: probe TemporalTransformer's attention internals across a set of
stage2 checkpoints, on a real video clip.

TT's heads are no longer uniform (model/temporal.py), so this script reports two
different things depending on head type, not one uniform per-head table:
  - QK+ALiBi heads (num_alibi_heads of them): the two additive terms that make up
    their pre-softmax logits - QK^T/sqrt(head_dim) (content attention) and
    -n_h * |i-j| (ALiBi-style distance bias, Sec 4.4).
  - The single visibility-only head: its logits ARE the raw per-key-frame
    visibility score directly (Sec 4.4) - no QK term, no distance term, so
    there's nothing to compare it against within its own softmax; it's reported
    separately, not folded into the QK+ALiBi table.
Also measures how far TT's output has actually moved from doing nothing:
output_proj is zero-initialized (model/temporal.py), so TT(tokens) == tokens
exactly until output_proj's weight/bias move away from zero during training.

Usage:
    python scripts/analyze_tt_attention.py \
        [--checkpoint_dir /projects/u6ga/sk3925_misc/checkpoints/stage2_AC] \
        [--input_path <frame_dir>]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import COMPONENT_TOKENS  # noqa: E402
from model.encoding import _decode_params, _fill_missing_frame_tokens  # noqa: E402
from model.temporal import _gather_local_windows  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    FrameJob,
    build_models,
    load_available_checkpoint,
    make_crop_parse_pool,
    run_parallel_crop_and_parse,
)

DEFAULT_CLIP = (
    "/projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/"
    "PHOENIX-2014-T/features/fullFrame-210x260px/test/01April_2011_Friday_tagesschau-3377"
)
DEFAULT_CHECKPOINT_DIR = "/projects/u6ga/sk3925_misc/checkpoints/stage2_AC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--input_path", type=str, default=DEFAULT_CLIP)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--xseg_device", type=str, default=default_device)
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1))
    return parser.parse_args()


def _checkpoint_paths(checkpoint_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not paths:
        raise ValueError(f"No checkpoints found under {checkpoint_dir}")
    return sorted(paths, key=lambda p: int(re.search(r"step_(\d+)\.pt$", p).group(1)))


def _load_real_clip(args: argparse.Namespace):
    """-> (clip_pixel_values (1,N,3,H,W), visibility_scores (1,N),
    frame_indices (1,N), flag_visibility_valid (1,N), real_frame_mask (1,N)),
    all on args.device. Mirrors inference/demo_videos.py's own frame-loading
    block (--image_seq path) - same crop/XSeg pipeline, just without any
    rendering."""
    frames, _fps, video_name = load_frames(args.input_path, image_seq=True, fps=30)
    if not frames:
        raise ValueError(f"No frames found at {args.input_path}")
    num_frames = len(frames)

    jobs = [FrameJob(video_name, i, frame_bgr, args.crop_scale, args.image_size) for i, frame_bgr in enumerate(frames)]
    executor = make_crop_parse_pool(args.num_workers)
    try:
        results = run_parallel_crop_and_parse(jobs, executor, args.xseg_device)
    finally:
        executor.shutdown()
    results_by_frame = {result.frame_index: result for result in results}

    pixel_values_list, visibility_list, valid_list = [], [], []
    for i in range(num_frames):
        result = results_by_frame[i]
        frame_valid = result.valid and result.cropped_rgb is not None
        if frame_valid:
            pixel_values = torch.from_numpy(result.cropped_rgb).permute(2, 0, 1).float().to(args.device) / 255.0
        else:
            pixel_values = torch.zeros(3, args.image_size, args.image_size, device=args.device)
        pixel_values_list.append(pixel_values)
        visibility_list.append(result.visibility_ratio)
        valid_list.append(frame_valid)

    invalid_count = valid_list.count(False)
    clip_pixel_values = torch.stack(pixel_values_list, dim=0).unsqueeze(0)
    visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=args.device).unsqueeze(0)
    frame_indices = torch.arange(num_frames, device=args.device).float().unsqueeze(0)
    flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=args.device).unsqueeze(0)
    real_frame_mask = torch.ones(1, num_frames, dtype=torch.bool, device=args.device)

    print(
        f"[analyze_tt] loaded {num_frames} frames from {args.input_path} "
        f"(visibility range {min(visibility_list):.3f}-{max(visibility_list):.3f}, "
        f"{invalid_count} face-detection failures)"
    )
    return clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid, real_frame_mask


def _build_tokens(svit, clip_pixel_values, real_frame_mask, flag_visibility_valid):
    """SViT-encode every frame independently (pre-TT tokens), matching
    model.encoding.encode_video's own token-construction block exactly, so the
    no-TT ("tokens") and with-TT ("refined") comparisons below are on the same
    footing encode_video itself would produce."""
    batch_size, num_frames = clip_pixel_values.shape[:2]
    flat_pixel_values = clip_pixel_values.reshape(batch_size * num_frames, *clip_pixel_values.shape[2:])
    with torch.no_grad():
        component_outputs = svit(flat_pixel_values)
    component_names = [t.name for t in COMPONENT_TOKENS]
    tokens = torch.stack(
        [component_outputs[name].reshape(batch_size, num_frames, -1) for name in component_names], dim=2
    )
    tokens = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    return tokens, component_names


def _key_valid_mask(real_frame_mask: torch.Tensor, radius: int, num_components: int) -> torch.Tensor:
    """(B, N) valid_mask -> (B*N, C, w*C) bool, True where a key slot is a real
    frame (matches TTBlock's B*N batching and distance_bias's/visibility_logits'
    window-major/component-minor key layout) - for filtering boundary/padding
    slots out of the magnitude stats below. Validity doesn't depend on head type,
    so this is shared by both the QK+ALiBi and visibility-only comparisons."""
    local_valid = _gather_local_windows(real_frame_mask, radius)  # (B, N, w)
    batch_size, num_frames, window_size = local_valid.shape
    token_valid = local_valid.repeat_interleave(num_components, dim=2)  # (B, N, w*C)
    token_valid = token_valid.unsqueeze(2).expand(-1, -1, num_components, -1)  # (B, N, C, w*C)
    return token_valid.reshape(batch_size * num_frames, num_components, -1)


@torch.no_grad()
def _run_tt_and_capture_normed(tt, tokens, visibility_scores, frame_indices, real_frame_mask):
    """Runs tt.forward once, capturing each block's post-norm1 activations
    (TTBlock.forward's own `normed`, the tensor its Q/K/V projections and
    _gather_local_windows both consume) via a forward hook, so the QK term
    can be reconstructed exactly afterward without a second forward pass."""
    normed_by_layer: list[list[torch.Tensor]] = []

    def make_hook(sink: list[torch.Tensor]):
        def hook(_module, _inputs, output):
            sink.append(output.detach())
        return hook

    hooks = []
    for block in tt.blocks:
        sink: list[torch.Tensor] = []
        hooks.append(block.norm1.register_forward_hook(make_hook(sink)))
        normed_by_layer.append(sink)

    refined = tt(tokens, visibility_scores, frame_indices, valid_mask=real_frame_mask)

    for h in hooks:
        h.remove()
    return refined, [sink[0] for sink in normed_by_layer]


@torch.no_grad()
def _bias_terms(tt, visibility_scores, frame_indices, real_frame_mask, num_components):
    """Distance bias (QK+ALiBi heads) and raw-visibility logits (the
    visibility-only head), computed via TemporalTransformer's own
    _compute_distance_bias/_compute_visibility_logits (Sec 4.3/4.4) rather than
    reimplementing that math here, reshaped to TTBlock's B*N batching so they
    line up entry-for-entry with _qk_logits' output. Also returns the shared
    key-validity mask (boundary/padding slots excluded from any magnitude
    comparison)."""
    radius = tt.config.window_size // 2
    batch_size, num_frames = visibility_scores.shape
    bn = batch_size * num_frames

    distance_bias = tt._compute_distance_bias(frame_indices, real_frame_mask, num_components)  # (B,N,Ha,C,w*C)
    visibility_logits = tt._compute_visibility_logits(visibility_scores, real_frame_mask, num_components)  # (B,N,C,w*C)
    distance_bias = distance_bias.reshape(bn, tt.num_alibi_heads, num_components, -1)
    visibility_logits = visibility_logits.reshape(bn, num_components, -1)

    key_valid = _key_valid_mask(real_frame_mask, radius, num_components)  # (bn, C, w*C)
    return distance_bias, visibility_logits, key_valid


@torch.no_grad()
def _qk_logits(block, normed: torch.Tensor, radius: int) -> torch.Tensor:
    """Reconstructs raw QK^T/sqrt(head_dim) for one TTBlock's QK+ALiBi heads,
    using the block's own q_proj/k_proj (model/temporal.py's TTBlock), from the
    same `normed` tensor TTBlock.forward feeds them (captured via the norm1
    hook) - the only one of the QK+ALiBi heads' two additive terms not
    otherwise directly observable (the visibility-only head has no QK term at
    all, see module docstring)."""
    batch_size, num_frames, num_components, dim = normed.shape
    window_size = 2 * radius + 1
    seq_len_kv = window_size * num_components

    kv = _gather_local_windows(normed, radius).reshape(batch_size * num_frames, seq_len_kv, dim)
    q = normed.reshape(batch_size * num_frames, num_components, dim)

    q_proj = block.q_proj(q).reshape(batch_size * num_frames, num_components, block.num_alibi_heads, block.head_dim)
    k_proj = block.k_proj(kv).reshape(batch_size * num_frames, seq_len_kv, block.num_alibi_heads, block.head_dim)
    q_proj = q_proj.permute(0, 2, 1, 3)
    k_proj = k_proj.permute(0, 2, 1, 3)
    return torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (block.head_dim**0.5)  # (B*N, Ha, C, w*C)


def _print_term_stats(step: int, tt, normed_by_layer, distance_bias, visibility_logits, key_valid) -> None:
    radius = tt.config.window_size // 2
    print(f"\n=== step {step}: attention-term magnitude, mean|x| (std) - per layer/head ===")
    print("QK+ALiBi heads (content attention + distance-only bias):")
    print(f"{'layer':>5} {'head':>4}   {'QK':>18} {'dist_bias':>18}")
    for layer_idx, normed in enumerate(normed_by_layer):
        block = tt.blocks[layer_idx]
        qk_logits = _qk_logits(block, normed, radius)  # (B*N, Ha, C, w*C)
        for head_idx in range(tt.num_alibi_heads):
            qk_vals = qk_logits[:, head_idx][key_valid]
            dist_vals = distance_bias[:, head_idx][key_valid]
            print(
                f"{layer_idx:>5} {head_idx:>4}   "
                f"{qk_vals.abs().mean().item():>8.4f} ({qk_vals.std().item():>6.4f})   "
                f"{dist_vals.abs().mean().item():>8.4f} ({dist_vals.std().item():>6.4f})"
            )

    # Not directly comparable to the QK+ALiBi table above: this head has no QK
    # term and no distance term, so its logit magnitude isn't "one term among
    # several" the way QK/dist_bias are - it's the whole thing.
    print("\nVisibility-only head (logits ARE the raw visibility score - no QK, no distance term):")
    vis_vals = visibility_logits[key_valid]
    print(f"  raw visibility logit: mean|x|={vis_vals.abs().mean().item():.4f} (std {vis_vals.std().item():.4f})")


def _print_output_effect(tt, heads, tokens, refined, component_names) -> None:
    delta = refined - tokens
    residual_norm = tokens.norm(dim=-1)
    delta_norm = delta.norm(dim=-1)
    ratio = (delta_norm / residual_norm.clamp(min=1e-8)).mean(dim=(0, 1))

    print(f"  output_proj weight norm: {tt.output_proj.weight.norm().item():.6f} (0.0 at init)")
    print(f"  output_proj bias norm:   {tt.output_proj.bias.norm().item():.6f} (0.0 at init)")
    print(f"  component_type_embedding norm: {tt.component_type_embedding.norm().item():.4f}")
    print("  mean ||delta|| / ||residual|| in token space, per component:")
    for name, r in zip(component_names, ratio.tolist()):
        print(f"    {name:>10}: {r:.6f}")

    with torch.no_grad():
        features_no_tt = {name: tokens[:, :, i, :] for i, name in enumerate(component_names)}
        features_with_tt = {name: refined[:, :, i, :] for i, name in enumerate(component_names)}
        params_no_tt = _decode_params(heads(features_no_tt))
        params_with_tt = _decode_params(heads(features_with_tt))

    print("  decoded FLAME/camera parameter diff (with-TT vs no-TT), mean ||diff|| per frame:")
    for key in params_no_tt:
        diff = (params_with_tt[key] - params_no_tt[key]).norm(dim=-1).mean()
        base = params_no_tt[key].norm(dim=-1).mean().clamp(min=1e-8)
        print(f"    {key:>10}: abs={diff.item():.6f}  rel={(diff / base).item():.6f}")


def main() -> None:
    args = parse_args()
    checkpoint_paths = _checkpoint_paths(args.checkpoint_dir)
    print(f"[analyze_tt] found {len(checkpoint_paths)} checkpoints: {[os.path.basename(p) for p in checkpoint_paths]}")

    clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid, real_frame_mask = _load_real_clip(args)

    models = build_models(args.device, use_unet=False)
    svit, tt, heads = models["svit"], models["tt"], models["heads"]

    for checkpoint_path in checkpoint_paths:
        step = load_available_checkpoint(models, checkpoint_path, args.device)
        tokens, component_names = _build_tokens(svit, clip_pixel_values, real_frame_mask, flag_visibility_valid)
        num_components = tokens.shape[2]

        print(f"\n{'=' * 70}\ncheckpoint: {checkpoint_path} (step {step})\n{'=' * 70}")
        refined, normed_by_layer = _run_tt_and_capture_normed(
            tt, tokens, visibility_scores, frame_indices, real_frame_mask
        )
        _print_output_effect(tt, heads, tokens, refined, component_names)

        distance_bias, visibility_logits, key_valid = _bias_terms(
            tt, visibility_scores, frame_indices, real_frame_mask, num_components
        )
        _print_term_stats(step, tt, normed_by_layer, distance_bias, visibility_logits, key_valid)


if __name__ == "__main__":
    main()

