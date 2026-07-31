"""Analyze per-frame face-visibility scores from the prewarmed face-parsing
cache for train-split videos from all 2d_video/3d_video datasets: csl_daily,
how2sign, phoenix2014t, afew_va, mead, coma, vocaset.

Reads visibility_ratio directly from the cached XSeg output (see
scripts/prewarm_face_parsing_cache.py) - no face detection/segmentation is
re-run here.

Usage:
    python scripts/analyze_visibility_scores.py
"""
import argparse
import csv
import io
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.config import load_dataloader_config  # noqa: E402
from dataset_processing.dataloading.frame_count_cache import load_frame_counts  # noqa: E402
from dataset_processing.dataloading.video_frames import make_frame_source  # noqa: E402
from dataset_processing.manifest_schema import read_manifest  # noqa: E402
from utils.cache_utils import bucket_container_path, entry_key, read_bucket_entry, sentinel_path  # noqa: E402

ALL_DATASETS = ["csl_daily", "how2sign", "phoenix2014t", "afew_va", "mead", "coma", "vocaset"]
MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_processing", "manifests")


def load_video_scores(face_parsing_cache_root, crop_cache_root, dataset, num_videos, seed):
    manifest_path = os.path.join(MANIFEST_DIR, f"{dataset}.jsonl")
    rows = [row for row in read_manifest(manifest_path) if row.split == "train"]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:num_videos]

    cached_counts = load_frame_counts(crop_cache_root, dataset) or {}

    videos = {}  # sample_id -> list of visibility_ratio floats
    missing_uncached = 0
    missing_noface = 0
    unreadable = 0
    for row in rows:
        num_frames = cached_counts.get(row.sample_id)
        if num_frames is None:
            try:
                source = make_frame_source(row.image_paths)
                num_frames = source.num_frames()
                source.close()
            except Exception:
                unreadable += 1
                continue

        container_path = bucket_container_path(face_parsing_cache_root, dataset, row.sample_id)
        scores = []
        for frame_index in range(num_frames):
            key = entry_key(row.sample_id, frame_index)
            data = read_bucket_entry(container_path, key)
            if data is None:
                noface_path = sentinel_path(face_parsing_cache_root, dataset, row.sample_id, frame_index, "noface")
                if noface_path.exists():
                    missing_noface += 1
                else:
                    missing_uncached += 1
                continue
            with np.load(io.BytesIO(data)) as npz:
                scores.append(float(npz["visibility_ratio"]))
        if scores:
            videos[row.sample_id] = scores
    return videos, missing_uncached, missing_noface, unreadable


def compute_stats(videos):
    all_scores = np.array([s for scores in videos.values() for s in scores])
    per_video_min = {sid: min(scores) for sid, scores in videos.items()}
    per_video_max = {sid: max(scores) for sid, scores in videos.items()}
    per_video_range = {sid: per_video_max[sid] - per_video_min[sid] for sid in videos}
    mean = float(all_scores.mean())
    worst_video = max(per_video_range, key=per_video_range.get)
    return {
        "num_videos": len(videos),
        "num_frames": int(all_scores.size),
        "mean_visibility": mean,
        "std_visibility": float(all_scores.std()),
        "min_visibility": float(all_scores.min()),
        "max_visibility": float(all_scores.max()),
        "max_deviation_below_mean": mean - float(all_scores.min()),
        "max_deviation_above_mean": float(all_scores.max()) - mean,
        "avg_min_visibility_per_video": float(np.mean(list(per_video_min.values()))),
        "avg_max_visibility_per_video": float(np.mean(list(per_video_max.values()))),
        "max_within_video_range": float(per_video_range[worst_video]),
        "video_with_max_range": worst_video,
        "video_with_max_range_min_score": float(per_video_min[worst_video]),
        "video_with_max_range_max_score": float(per_video_max[worst_video]),
        "avg_within_video_range": float(np.mean(list(per_video_range.values()))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_config", type=str, default="dataset_processing/config/dataloader.yaml")
    parser.add_argument("--num_videos", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/visibility_score_analysis")
    parser.add_argument("--datasets", type=str, nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    args = parser.parse_args()

    cfg = load_dataloader_config(args.dataloader_config)
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    for dataset in args.datasets:
        videos, missing_uncached, missing_noface, unreadable = load_video_scores(
            cfg.face_parsing_cache_root, cfg.crop_cache_root, dataset, args.num_videos, args.seed)

        csv_path = os.path.join(args.output_dir, f"{dataset}_scores.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "frame_index", "visibility_ratio"])
            for sample_id, scores in videos.items():
                for frame_index, score in enumerate(scores):
                    writer.writerow([sample_id, frame_index, score])

        stats = compute_stats(videos)
        stats["missing_uncached_frames"] = missing_uncached
        stats["missing_noface_frames"] = missing_noface
        stats["unreadable_videos"] = unreadable
        summary[dataset] = stats
        print(f"\n{dataset}: {json.dumps(stats, indent=2)}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved per-frame scores and {summary_path}")


if __name__ == "__main__":
    main()

