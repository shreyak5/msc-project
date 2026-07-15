import argparse
import csv
import functools
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.config import load_dataloader_config  # noqa: E402
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from dataset_processing.dataloading.face_parsing_cache import get_face_parsing  # noqa: E402
from dataset_processing.dataloading.face_parsing_pool import get_xseg  # noqa: E402
from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML, load_datasets_yaml  # noqa: E402
from dataset_processing.dataloading.video_frames import make_frame_source  # noqa: E402
from dataset_processing.manifest_schema import read_manifest  # noqa: E402

IMAGE_CATEGORIES = {"2d_image"}
# The face-region mask (masking -> UNet reconstruction, Sec 7 Pass A/C) only
# applies to 2D categories - matches dataset_processing/dataloading/datasets.py's
# CATEGORIES_2D. Visibility scores are different: TemporalTransformer needs them
# for every video category's clips regardless of 2D/3D, so 3d_video is prewarmed
# here too even though it never needs the mask half of this cache's output.
# 3d_image needs neither and is excluded.
PREWARM_CATEGORIES = {"2d_image", "2d_video", "3d_video"}


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
            rows = rows[args.shard_index::args.num_shards]
            is_image = entry.category in IMAGE_CATEGORIES

            for row in tqdm(rows, desc=entry.name):
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
                    if is_image or len(row.image_paths) == 1:
                        source_path = row.image_paths[0]
                    else:
                        source_path = row.image_paths[frame_index]

                    if is_image:
                        load_image = lambda p=source_path: cv2.imread(p)
                    else:
                        load_image = lambda fi=frame_index: source.read_frame(fi)

                    def on_noface(ds=entry.name, sid=row.sample_id, fi=frame_index, sp=source_path):
                        log_writer.writerow([ds, sid, fi if fi is not None else "", sp, "no_face_detected"])

                    def on_error(reason, ds=entry.name, sid=row.sample_id, fi=frame_index, sp=source_path):
                        log_writer.writerow([ds, sid, fi if fi is not None else "", sp, f"error: {reason}"])

                    try:
                        get_face_parsing(
                            cfg.face_parsing_cache_root, entry.name, row.sample_id, frame_index,
                            load_image, get_prewarm_detector, get_prewarm_xseg,
                            cfg.crop_scale, cfg.image_size,
                            on_noface=on_noface, on_error=on_error,
                        )
                    except Exception as exc:
                        # Defense-in-depth for anything unexpected that get_face_parsing's own
                        # on_error path doesn't already cover (e.g. a disk write failure) -
                        # the unreadable-source-image case is handled by on_error above and
                        # never reaches here.
                        on_error(str(exc))

                if source is not None:
                    source.close()

    print("DONE!")


if __name__ == "__main__":
    main()
