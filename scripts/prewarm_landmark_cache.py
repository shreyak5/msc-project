import argparse
import csv
import functools
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.config import load_dataloader_config  # noqa: E402
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from dataset_processing.dataloading.landmark_cache import compute_landmarks  # noqa: E402
from dataset_processing.dataloading.landmark_pool import get_fan_predictor, get_mediapipe_detector  # noqa: E402
from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML, load_datasets_yaml  # noqa: E402
from dataset_processing.dataloading.video_frames import make_frame_source  # noqa: E402
from dataset_processing.manifest_schema import read_manifest  # noqa: E402
from model.constants import MEDIAPIPE_TASK_MODEL_PATH  # noqa: E402
from utils.cache_utils import (  # noqa: E402
    bucket_container_path,
    entry_key,
    read_all_bucket_entries,
    shard_of,
    write_bucket_entries,
)

# The landmark loss (implementation-plan.md Sec 6/7) only applies to 2D
# batches - matches dataset_processing/dataloading/datasets.py's CATEGORIES_2D.
IMAGE_CATEGORIES = {"2d_image"}
CATEGORIES_2D = {"2d_image", "2d_video"}


def main():
    parser = argparse.ArgumentParser(
        description="Populate the GT landmark cache (FAN + MediaPipe) for 2D datasets before real training starts.")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--split", type=str, default="all", choices=["train", "dev", "test", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataloader_config", type=str, required=True)
    parser.add_argument("--datasets_yaml", type=str, default=str(DEFAULT_DATASETS_YAML))
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_dataloader_config(args.dataloader_config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device={args.device} requested but torch.cuda.is_available() is False "
            f"(SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) - "
            "this shard has no GPU bound. Check srun's own --gres flag rather than "
            "letting every row silently fail (see slurm_jobs/prewarm_landmark_cache.sh)."
        )
    entries = [e for e in load_datasets_yaml(args.datasets_yaml) if e.category in CATEGORIES_2D]
    if args.dataset != "all":
        entries = [entry for entry in entries if entry.name == args.dataset]

    get_prewarm_detector = functools.partial(
        get_detector, args.device, cfg.detector.threshold, cfg.detector.model_name)
    get_prewarm_fan_predictor = functools.partial(get_fan_predictor, args.device, "2dfan4")
    get_prewarm_mediapipe_detector = functools.partial(get_mediapipe_detector, MEDIAPIPE_TASK_MODEL_PATH)

    log_path = Path(cfg.landmark_cache_root) / "prewarm_logs" / f"shard_{args.shard_index}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["dataset", "sample_id", "frame_index", "source_path", "reason"])

        for entry in entries:
            rows = [
                row for row in read_manifest(entry.manifest_path)
                if args.split == "all" or row.split == args.split
            ]
            # Hash-bucket ownership instead of positional striping: shard_of
            # uses the same digest bytes as a sample's bucket path itself, so
            # every sample_id in a given bucket is always owned by the same
            # shard - no two concurrent shards ever write the same bucket file.
            rows = [row for row in rows if shard_of(row.sample_id, args.num_shards) == args.shard_index]
            is_image = entry.category in IMAGE_CATEGORIES

            # Group this shard's owned rows by bucket so each bucket's
            # container is read once and written once per run (see
            # write_bucket_entries), not once per frame.
            rows_by_bucket: dict[Path, list] = defaultdict(list)
            for row in rows:
                container_path = bucket_container_path(cfg.landmark_cache_root, entry.name, row.sample_id)
                rows_by_bucket[container_path].append(row)

            for container_path, bucket_rows in tqdm(rows_by_bucket.items(), desc=entry.name):
                try:
                    existing_entries = read_all_bucket_entries(container_path)
                except Exception:
                    # A corrupted bucket here just means "nothing to skip" -
                    # every frame below gets recomputed, and the eventual
                    # write_bucket_entries call is what actually rebuilds it.
                    existing_entries = {}
                new_entries: dict[str, bytes] = {}
                new_entries_meta: dict[str, tuple[str, "int | None", str]] = {}

                for row in bucket_rows:
                    source = None
                    if not is_image:
                        try:
                            source = make_frame_source(row.image_paths)
                            num_frames = source.num_frames()
                        except Exception as exc:
                            log_writer.writerow([
                                entry.name, row.sample_id, "", row.image_paths[0], f"unreadable_video: {exc}",
                            ])
                            continue
                    frame_indices = [None] if is_image else range(num_frames)

                    for frame_index in frame_indices:
                        key = entry_key(row.sample_id, frame_index)
                        if key in existing_entries:
                            continue  # already cached from a previous run

                        if is_image or len(row.image_paths) == 1:
                            source_path = row.image_paths[0]
                        else:
                            source_path = row.image_paths[frame_index]

                        if is_image:
                            load_image = lambda p=source_path: cv2.imread(p)
                        else:
                            load_image = lambda fi=frame_index: source.read_frame(fi)

                        try:
                            result = compute_landmarks(
                                entry.name, row.sample_id, frame_index,
                                load_image, get_prewarm_detector,
                                get_prewarm_fan_predictor, get_prewarm_mediapipe_detector,
                                cfg.crop_cache_root, cfg.crop_scale, cfg.image_size,
                            )
                        except Exception as exc:
                            # Defense-in-depth for anything unexpected compute_landmarks'
                            # own status handling doesn't already cover.
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index if frame_index is not None else "",
                                source_path, f"error: {exc}",
                            ])
                            continue

                        if result.status == "unreadable":
                            # Deliberately NOT persisted (see LandmarkResult's docstring) -
                            # the next run will retry crop_cache rather than replay a
                            # cached transient failure.
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index if frame_index is not None else "",
                                source_path, f"error: {result.error}",
                            ])
                            continue

                        # compute_landmarks has no on_noface-style signal (unlike
                        # compute_mica_shape) - a missing face is only visible via the
                        # returned flags, so log it here rather than inside the cache
                        # function itself. Still persisted as a regular entry (all-False
                        # flags), matching get_landmarks' own pre-existing behavior.
                        if not result.flag_landmarks_fan_valid and not result.flag_landmarks_mp_valid:
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index if frame_index is not None else "",
                                source_path, "no_face_detected",
                            ])

                        buffer = io.BytesIO()
                        np.savez(
                            buffer,
                            landmarks_fan=result.landmarks_fan, flag_landmarks_fan_valid=result.flag_landmarks_fan_valid,
                            landmarks_mp=result.landmarks_mp, flag_landmarks_mp_valid=result.flag_landmarks_mp_valid,
                        )
                        new_entries[key] = buffer.getvalue()
                        new_entries_meta[key] = (row.sample_id, frame_index, source_path)

                    if source is not None:
                        source.close()

                if new_entries:
                    try:
                        write_bucket_entries(
                            cfg.landmark_cache_root, entry.name, bucket_rows[0].sample_id, new_entries)
                    except Exception as exc:
                        # The whole batch for this bucket failed to persist (e.g. disk
                        # quota) - attribute it to every frame that would have been in it,
                        # matching the per-frame granularity the log already used before
                        # this bucket-batched rewrite.
                        for key, (sample_id, frame_index, source_path) in new_entries_meta.items():
                            log_writer.writerow([
                                entry.name, sample_id, frame_index if frame_index is not None else "",
                                source_path, f"error: {exc}",
                            ])

    print("DONE!")


if __name__ == "__main__":
    main()
