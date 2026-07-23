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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.config import load_dataloader_config  # noqa: E402
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from dataset_processing.dataloading.face_parsing_cache import compute_face_parsing  # noqa: E402
from dataset_processing.dataloading.face_parsing_pool import get_xseg  # noqa: E402
from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML, load_datasets_yaml  # noqa: E402
from dataset_processing.dataloading.video_frames import make_frame_source  # noqa: E402
from dataset_processing.manifest_schema import read_manifest  # noqa: E402
from utils.cache_utils import (  # noqa: E402
    atomic_write_bytes,
    bucket_container_path,
    entry_key,
    read_all_bucket_entries,
    sentinel_path,
    shard_of,
    write_bucket_entries,
)

IMAGE_CATEGORIES = {"2d_image", "3d_image"}
# face_mask is needed for all 4 categories (Pass A/C's masking -> UNet
# reconstruction touches 2D batches only, but Pass B explicitly treats 3D
# datasets' images as generic 2D images too - "3D datasets contribute their 2D
# images, meshes ignored" - see dataset_processing/dataloading/datasets.py's
# build_category_dataset comment for the full reasoning). visibility_ratio is
# only meaningful for the two video categories (TemporalTransformer, Pass C),
# but since both fields come from one XSeg call, prewarming all 4 categories
# together is simpler than trying to prewarm the two fields separately.
PREWARM_CATEGORIES = {"2d_image", "2d_video", "3d_image", "3d_video"}


def main():
    parser = argparse.ArgumentParser(
        description="Populate the face-parsing cache (XSeg mask + visibility ratio) before real training starts.")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--split", type=str, default="all", choices=["train", "dev", "test", "all"])
    # Only accelerates the RetinaFace detection half of this script - XSeg
    # itself runs through onnxruntime (face_parsing_pool.get_xseg), which has
    # no CUDA execution provider available on this cluster's aarch64 nodes
    # (onnxruntime-gpu has no PyPI wheel there - see dataloader.yaml's
    # xseg_device comment), so XSeg parsing is CPU-bound regardless of this flag.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataloader_config", type=str, required=True)
    parser.add_argument("--datasets_yaml", type=str, default=str(DEFAULT_DATASETS_YAML))
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_dataloader_config(args.dataloader_config)
    entries = [e for e in load_datasets_yaml(args.datasets_yaml) if e.category in PREWARM_CATEGORIES]
    if args.dataset != "all":
        entries = [entry for entry in entries if entry.name == args.dataset]

    get_prewarm_detector = functools.partial(
        get_detector, args.device, cfg.detector.threshold, cfg.detector.model_name)
    get_prewarm_xseg = functools.partial(get_xseg, args.device)

    log_path = Path(cfg.face_parsing_cache_root) / "prewarm_logs" / f"shard_{args.shard_index}.csv"
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
                container_path = bucket_container_path(cfg.face_parsing_cache_root, entry.name, row.sample_id)
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
                        noface_path = sentinel_path(
                            cfg.face_parsing_cache_root, entry.name, row.sample_id, frame_index, "noface")
                        unreadable_path = sentinel_path(
                            cfg.face_parsing_cache_root, entry.name, row.sample_id, frame_index, "unreadable")
                        if key in existing_entries or noface_path.exists() or unreadable_path.exists():
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
                            result = compute_face_parsing(
                                load_image, get_prewarm_detector, get_prewarm_xseg,
                                cfg.crop_scale, cfg.image_size,
                            )
                        except Exception as exc:
                            # Defense-in-depth for anything unexpected compute_face_parsing's
                            # own status handling doesn't already cover.
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index if frame_index is not None else "",
                                source_path, f"error: {exc}",
                            ])
                            continue

                        if result.status == "unreadable":
                            atomic_write_bytes(unreadable_path, b"")
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index if frame_index is not None else "",
                                source_path, f"error: {result.error}",
                            ])
                            continue

                        if result.status == "noface":
                            atomic_write_bytes(noface_path, b"")
                            log_writer.writerow([
                                entry.name, row.sample_id, frame_index if frame_index is not None else "",
                                source_path, "no_face_detected",
                            ])
                            continue

                        buffer = io.BytesIO()
                        np.savez(
                            buffer, face_mask=result.face_mask,
                            visibility_ratio=np.float32(result.visibility_ratio),
                        )
                        new_entries[key] = buffer.getvalue()
                        new_entries_meta[key] = (row.sample_id, frame_index, source_path)

                    if source is not None:
                        source.close()

                if new_entries:
                    try:
                        write_bucket_entries(
                            cfg.face_parsing_cache_root, entry.name, bucket_rows[0].sample_id, new_entries)
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
