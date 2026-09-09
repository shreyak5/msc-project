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
    window_indices = sorted(i for i in vis_by_frame if start <= i < end)
    for a, b in zip(window_indices, window_indices[1:]):
        if b == a + 1 and abs(vis_by_frame[b] - vis_by_frame[a]) >= cap:
            return True
    return False

