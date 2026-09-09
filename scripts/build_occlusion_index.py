from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.config import load_dataloader_config  # noqa: E402
from dataset_processing.dataloading.datasets import _rows_with_frame_counts  # noqa: E402
from dataset_processing.dataloading.occlusion_index import is_window_occlusion_positive  # noqa: E402
from dataset_processing.dataloading.video_frames import segment_starts  # noqa: E402
from utils.cache_utils import bucket_container_path, entry_key  # noqa: E402

ALL_DATASETS = ["csl_daily", "how2sign", "phoenix2014t"]
MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_processing", "manifests")


def read_visibility_by_frame(
    face_parsing_cache_root: str, dataset: str, sample_id: str, num_frames_total: int,
) -> dict[int, float]:
    container_path = bucket_container_path(face_parsing_cache_root, dataset, sample_id)
    if not container_path.exists():
        return {}
    result: dict[int, float] = {}
    with zipfile.ZipFile(container_path, "r") as zf:
        names = set(zf.namelist())
        for frame_index in range(num_frames_total):
            key = entry_key(sample_id, frame_index)
            if key not in names:
                continue
            with zf.open(key) as f:
                data = f.read()
            with np.load(io.BytesIO(data)) as npz:
                result[frame_index] = float(npz["visibility_ratio"])
    return result


def build_index_for_dataset(
    dataset: str, cfg, max_frames: int, cap: float, split: str, progress_interval: int = 500,
) -> list[dict[str, int | str]]:
    """progress_interval: print a running count every this many videos scanned
    (0 disables). flush=True on every print here since stdout is block-
    buffered (not line-buffered) once redirected to a file/sbatch log rather
    than a real terminal - without it, nothing would actually appear in the
    log until the whole process exits, defeating the point of progress
    output for a long-running job."""
    manifest_path = os.path.join(MANIFEST_DIR, f"{dataset}.jsonl")
    rows = _rows_with_frame_counts(dataset, manifest_path, split, cfg.crop_cache_root)
    print(f"{dataset}: {len(rows)} train videos to scan", flush=True)

    entries: list[dict[str, int | str]] = []
    for i, (row, num_frames_total) in enumerate(rows):
        vis_by_frame = read_visibility_by_frame(
            cfg.face_parsing_cache_root, dataset, row.sample_id, num_frames_total,
        )
        for start in segment_starts(num_frames_total, max_frames):
            end = min(start + max_frames, num_frames_total)
            if is_window_occlusion_positive(vis_by_frame, start, end, cap):
                entries.append({"sample_id": row.sample_id, "start": start})
        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            print(
                f"{dataset}: {i + 1}/{len(rows)} videos scanned, "
                f"{len(entries)} occlusion-positive windows so far", flush=True,
            )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataloader_config", type=str, default="dataset_processing/config/dataloader.yaml")
    parser.add_argument("--datasets", type=str, nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--cap", type=float, default=0.1,
        help="Occlusion-event threshold on |Δvisibility_ratio| between temporally adjacent "
        "cached frames - should match training/config.py's Stage2Config.vertex_gate_delta_cap "
        "so the offline subset and the training-time gate agree on what counts as an "
        "occlusion event.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Window size - defaults to the dataloader config's 2d_video category max_frames "
        "(what Pass C's clip_loader actually uses), so the resulting (sample_id, start) pairs "
        "line up with VideoFaceDataset's own segment_starts calls exactly.",
    )
    parser.add_argument("--output_dir", type=str, default="dataset_processing/occlusion_clip_index")
    parser.add_argument(
        "--progress_interval", type=int, default=500,
        help="Print a running count every this many videos scanned (0 disables).",
    )
    args = parser.parse_args()

    cfg = load_dataloader_config(args.dataloader_config)
    max_frames = args.max_frames if args.max_frames is not None else cfg.categories["2d_video"].max_frames
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        entries = build_index_for_dataset(dataset, cfg, max_frames, args.cap, args.split, args.progress_interval)
        out_path = os.path.join(args.output_dir, f"{dataset}.jsonl")
        with open(out_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        total_videos = len({entry["sample_id"] for entry in entries})
        print(
            f"{dataset}: {len(entries)} occlusion-positive windows across {total_videos} videos -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

