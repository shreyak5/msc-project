"""Occlusion-window classification for Pass C's occlusion-only clip-subset
experiment (training/config.py's Stage2Config.pass_c_occlusion_subset_index_dir,
scripts/build_occlusion_index.py). Kept torch-free (plain Python/dicts) since
this only ever runs offline, well before any model is built - the resulting
index is consumed at dataloader-construction time via
dataset_processing/dataloading/combined_loader.py."""

from __future__ import annotations


def is_window_occlusion_positive(
    vis_by_frame: dict[int, float], start: int, end: int, cap: float,
) -> bool:
    """vis_by_frame: {frame_index: visibility_ratio} for one video - only
    cached (detected) frames are present; a missing frame_index means that
    frame had no face-parsing cache entry (uncached or no-face), same
    convention scripts/analyze_visibility_scores.py's own cache read uses.

    start/end: the window's real frame-index range [start, end) - e.g.
    dataset_processing/dataloading/video_frames.py's segment_starts for
    `start`, with end = min(start + max_frames, num_frames_total). This is
    deliberately the UNPADDED range - not the max_frames-wide range
    frame_indices_for_segment produces for a short tail window, which repeats
    the last real frame as padding (a repeat trivially has zero diff, so
    including it wouldn't create a false positive, but there is no reason to
    look past the real frames either).

    Returns True iff any two TEMPORALLY ADJACENT cached frames within
    [start, end) (i.e. consecutive frame_index values that are BOTH cached -
    not merely the next cached entry after a gap) have
    |vis_by_frame[a] - vis_by_frame[b]| >= cap. This is the exact same
    "occlusion event" definition as model/losses/temporal_smoothness.py's
    compute_vertex_gate(mode="delta_vis") uses at training time (raw
    frame-to-frame difference, normalized against the same cap by default),
    so the offline subset and the training-time gate agree on what counts as
    an occlusion event."""
    window_indices = sorted(i for i in vis_by_frame if start <= i < end)
    for a, b in zip(window_indices, window_indices[1:]):
        if b == a + 1 and abs(vis_by_frame[b] - vis_by_frame[a]) >= cap:
            return True
    return False

