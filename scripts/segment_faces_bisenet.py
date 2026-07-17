"""Segment a face image (or directory of face images) into facial components
using UniFace's BiSeNet face parser.

Usage:
    python scripts/segment_faces_bisenet.py --input_path <face_image_or_dir>
"""
import argparse
import json
import os

import cv2
import numpy as np
from uniface.draw import FACE_PARSING_LABELS, vis_parsing_maps
from uniface.parsing import BiSeNet

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def is_image_file(filepath):
    return os.path.splitext(filepath)[1].lower() in VALID_IMAGE_EXTENSIONS


def gather_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if is_image_file(f))
        return [os.path.join(input_path, f) for f in files]
    raise ValueError(f"input_path '{input_path}' is not a file or directory")


def summarize_mask(mask):
    total_pixels = mask.size
    class_ids, counts = np.unique(mask, return_counts=True)
    return {
        FACE_PARSING_LABELS[class_id]: {
            'pixel_count': int(count),
            'pixel_fraction': float(count) / total_pixels,
        }
        for class_id, count in zip(class_ids.tolist(), counts.tolist())
    }


def main():
    parser_args = argparse.ArgumentParser(description='Segment face images into facial components with UniFace BiSeNet.')
    parser_args.add_argument('--input_path', type=str, required=True, help='Path to a (cropped) face image or a directory of face images')
    parser_args.add_argument('--output_dir', type=str, default='output/face_parsing', help='Directory to save overlays, masks, and stats')
    args = parser_args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    face_parser = BiSeNet()

    image_paths = gather_image_paths(args.input_path)
    if not image_paths:
        raise ValueError(f'No images found at {args.input_path}')

    results = {}
    num_skipped = 0

    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path)
        if image is None:
            print(f'[skip] Failed to load image: {path}')
            results[name] = None
            num_skipped += 1
            continue

        mask = face_parser.parse(image)
        vis_result = vis_parsing_maps(image, mask, save_image=False)

        cv2.imwrite(os.path.join(args.output_dir, f'{name}_parsing.png'), vis_result)
        np.save(os.path.join(args.output_dir, f'{name}_mask.npy'), mask)

        component_stats = summarize_mask(mask)
        results[name] = component_stats
        print(f'[ok] {name}: {len(component_stats)} facial components detected')

    stats_path = os.path.join(args.output_dir, 'segmentation_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2)

    num_processed = len(image_paths) - num_skipped
    print(f'\nProcessed {num_processed}/{len(image_paths)} images. Skipped {num_skipped}.')
    print(f'Saved overlays, masks, and {stats_path}')


if __name__ == '__main__':
    main()
