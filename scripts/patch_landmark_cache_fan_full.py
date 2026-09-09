import argparse
import csv
import functools
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

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

DEFAULT_DATASETS = "how2sign,phoenix2014t,csl_daily"


def main():
    parser = argparse.ArgumentParser(
        description="Patch the GT landmark cache with full 68-point FAN landmarks "
                    "for a dataset's dev-split entries (additive - preserves existing fields).")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASETS,
                        help="Comma-separated dataset names (default: the 3 sign-language dev-eval datasets).")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test", "all"])
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
            "letting every row silently fail (see slurm_jobs/patch_landmark_cache_fan_full.sh)."
        )
    dataset_names = {name.strip() for name in args.dataset.split(",")}
    entries = [e for e in load_datasets_yaml(args.datasets_yaml) if e.name in dataset_names]

    get_patch_detector = functools.partial(
        get_detector, args.device, cfg.detector.threshold, cfg.detector.model_name)
    get_patch_fan_predictor = functools.partial(get_fan_predictor, args.device, "2dfan4")
    get_patch_mediapipe_detector = functools.partial(get_mediapipe_detector, MEDIAPIPE_TASK_MODEL_PATH)

    log_path = Path(cfg.landmark_cache_root) / "patch_fan_full_logs" / f"shard_{args.shard_index}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["dataset", "sample_id", "frame_index", "source_path", "reason"])

        for entry in entries:
            rows = [
                row for row in read_manifest(entry.manifest_path)
                if args.split == "all" or row.split == args.split
            ]
            # Hash-bucket ownership, same rule prewarm_landmark_cache.py uses: every
            # sample_id in a given bucket is always owned by the same shard, so no
            # two concurrent shards ever touch the same bucket file.
            rows = [row for row in rows if shard_of(row.sample_id, args.num_shards) == args.shard_index]

            rows_by_bucket: dict[Path, list] = defaultdict(list)
            for row in rows:
                container_path = bucket_container_path(cfg.landmark_cache_root, entry.name, row.sample_id)
                rows_by_bucket[container_path].append(row)

            for container_path, bucket_rows in tqdm(rows_by_bucket.items(), desc=entry.name):
                try:
                    existing_entries = read_all_bucket_entries(container_path)
                except Exception:
                    existing_entries = {}
                patched_entries: dict[str, bytes] = {}
                patched_entries_meta: dict[str, tuple[str, int, str]] = {}

                for row in bucket_rows:
                    try:
                        source = make_frame_source(row.image_paths)
                        num_frames = source.num_frames()
                    except Exception as exc:
                        log_writer.writerow([
                            entry.name, row.sample_id, "", row.image_paths[0], f"unreadable_video: {exc}",
                        ])
                        continue

                    for frame_index in range(num_frames):
                        key = entry_key(row.sample_id, frame_index)
                        source_path = row.image_paths[0] if len(row.image_paths) == 1 else row.image_paths[frame_index]

                        if key not in existing_entries:
                            # Expected to never fire for the 3 dev-eval datasets (their
                            # dev-split base cache is already fully populated) - a
                            # defensive guard, not a normal code path.
                            log_writer.writerow([entry.name, row.sample_id, frame_index, source_path, "missing_base_entry"])
                            continue

                        cached = np.load(io.BytesIO(existing_entries[key]))
                        if "landmarks_fan_full" in cached.files:
                            continue  # already patched - idempotent re-run/resume

                        try:
                            result = compute_landmarks(
                                entry.name, row.sample_id, frame_index,
                                lambda fi=frame_index: source.read_frame(fi), get_patch_detector,
                                get_patch_fan_predictor, get_patch_mediapipe_detector,
                                cfg.crop_cache_root, cfg.crop_scale, cfg.image_size,
                                include_fan_full=True,
                            )
                        except Exception as exc:
                            log_writer.writerow([entry.name, row.sample_id, frame_index, source_path, f"error: {exc}"])
                            continue

                        if result.status == "unreadable":
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index, source_path, f"error: {result.error}",
                            ])
                            continue

                        # Preserve every pre-existing field byte-for-byte, add only the
                        # two new ones - never a bare write of just the new fields.
                        merged = dict(cached)
                        merged["landmarks_fan_full"] = result.landmarks_fan_full
                        merged["flag_landmarks_fan_full_valid"] = result.flag_landmarks_fan_full_valid
                        buffer = io.BytesIO()
                        np.savez(buffer, **merged)
                        patched_entries[key] = buffer.getvalue()
                        patched_entries_meta[key] = (row.sample_id, frame_index, source_path)

                    source.close()

                if patched_entries:
                    try:
                        write_bucket_entries(cfg.landmark_cache_root, entry.name, bucket_rows[0].sample_id, patched_entries)
                    except Exception as exc:
                        # The whole batch for this bucket failed to persist (e.g. disk
                        # quota) - attribute it to every frame that would have been in
                        # it, using the per-key metadata rather than a placeholder.
                        for key, (sample_id, frame_index, source_path) in patched_entries_meta.items():
                            log_writer.writerow([entry.name, sample_id, frame_index, source_path, f"error: {exc}"])

    print("DONE!")


if __name__ == "__main__":
    main()
