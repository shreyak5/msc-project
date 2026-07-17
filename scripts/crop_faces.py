"""Crop faces out of an image or a directory of images using RetinaFace.

Usage:
    python scripts/crop_faces.py --input_path <image_or_dir>
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cropping import build_retinaface_detector, crop_face  # noqa: E402
from preprocessing.io import is_image_file  # noqa: E402


def gather_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if is_image_file(f))
        return [os.path.join(input_path, f) for f in files]
    raise ValueError(f"input_path '{input_path}' is not a file or directory")


def main():
    parser = argparse.ArgumentParser(description='Crop faces out of an image or directory of images with RetinaFace.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the face detector on')
    parser.add_argument('--scale', type=float, default=1.4, help='Crop scale factor relative to the detected face box')
    parser.add_argument('--crop_size', type=int, default=224, help='Output crop size (square)')
    parser.add_argument('--output_dir', type=str, default='output/cropped_faces', help='Directory to save cropped images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    face_detector = build_retinaface_detector(args.device)

    image_paths = gather_image_paths(args.input_path)
    if not image_paths:
        raise ValueError(f'No images found at {args.input_path}')

    num_skipped = 0
    for path in image_paths:
        name = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            print(f'[skip] Failed to load image: {path}')
            num_skipped += 1
            continue

        cropped, _tform = crop_face(image, face_detector, scale=args.scale, image_size=args.crop_size)
        if cropped is None:
            print(f'[skip] No face detected: {path}')
            num_skipped += 1
            continue

        cv2.imwrite(os.path.join(args.output_dir, name), cropped)
        print(f'[ok] {name}')

    num_processed = len(image_paths) - num_skipped
    print(f'\nProcessed {num_processed}/{len(image_paths)} images. Skipped {num_skipped}.')
    print(f'Saved crops to {args.output_dir}')


if __name__ == '__main__':
    main()
