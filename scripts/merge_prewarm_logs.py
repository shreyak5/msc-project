import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.frame_count_cache import merge_frame_count_shards  # noqa: E402
from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML, load_datasets_yaml  # noqa: E402


def merge_logs(crop_cache_root: str, num_shards: int) -> None:
    logs_dir = Path(crop_cache_root) / "prewarm_logs"
    fieldnames = ["dataset", "sample_id", "frame_index", "source_path", "reason"]
    merged_rows = []
    for i in range(num_shards):
        shard_path = logs_dir / f"shard_{i}.csv"
        if not shard_path.exists():
            continue
        with open(shard_path, newline="") as f:
            reader = csv.DictReader(f)
            merged_rows.extend(reader)

    merged_path = logs_dir / "merged.csv"
    with open(merged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Merged {num_shards} shard logs ({len(merged_rows)} rows) into {merged_path}")


def merge_frame_counts(crop_cache_root: str, num_shards: int, datasets_yaml: str) -> None:
    for entry in load_datasets_yaml(datasets_yaml):
        merged = merge_frame_count_shards(crop_cache_root, entry.name, num_shards)
        if merged:
            print(f"Merged {num_shards} frame-count shards ({len(merged)} videos) for {entry.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Finalize a sharded prewarm run: merge per-shard no-face/error logs into one "
                    "CSV, and (for the crop cache only) merge per-shard video frame-count caches "
                    "into one file per dataset - prewarm_mica_cache.py probes frame counts live "
                    "itself rather than caching them, so there's no MICA-side shard to merge there.")
    parser.add_argument("--crop_cache_root", type=str, default=None)
    parser.add_argument("--mica_cache_root", type=str, default=None)
    parser.add_argument("--landmark_cache_root", type=str, default=None)
    parser.add_argument("--face_parsing_cache_root", type=str, default=None)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--datasets_yaml", type=str, default=str(DEFAULT_DATASETS_YAML))
    args = parser.parse_args()

    if args.crop_cache_root is not None:
        merge_logs(args.crop_cache_root, args.num_shards)
        merge_frame_counts(args.crop_cache_root, args.num_shards, args.datasets_yaml)
    if args.mica_cache_root is not None:
        merge_logs(args.mica_cache_root, args.num_shards)
    if args.landmark_cache_root is not None:
        merge_logs(args.landmark_cache_root, args.num_shards)
    if args.face_parsing_cache_root is not None:
        merge_logs(args.face_parsing_cache_root, args.num_shards)


if __name__ == "__main__":
    main()
