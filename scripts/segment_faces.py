"""Segment faces in an image or directory of images using uniface's XSeg model
and visualize the resulting mask (original / mask / overlay), following
uniface/examples/09_face_segmentation.ipynb.

Usage:
    python scripts/segment_faces.py --input_path <image_or_dir>
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from uniface.detection import RetinaFace
from uniface.parsing import XSeg

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def is_image_file(filepath):
    return os.path.splitext(filepath)[1].lower() in VALID_IMAGE_EXTENSIONS


def gather_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if is_image_file(f))
        return [os.path.join(input_path, f) for f in files]
    raise ValueError(f"input_path '{input_path}' is not a file or directory")


def segment_image(image, detector, parser):
    """Detect faces and accumulate an XSeg mask across all detected faces."""
    faces = detector.detect(image)
    full_mask = np.zeros(image.shape[:2], dtype=np.float32)
    num_segmented = 0
    for face in faces:
        if face.landmarks is None:
            continue
        mask = parser.parse(image, landmarks=face.landmarks)
        full_mask = np.maximum(full_mask, mask)
        num_segmented += 1
    return full_mask, faces, num_segmented


def apply_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """Apply colored mask overlay on image (BGR), matching the notebook's helper."""
    overlay = image.copy().astype(np.float32)
    color_overlay = np.zeros_like(image, dtype=np.float32)
    color_overlay[:] = color
    mask_3ch = mask[..., np.newaxis]
    overlay = overlay * (1 - mask_3ch * alpha) + color_overlay * mask_3ch * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def save_visualization(image, mask, overlay, out_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title('Mask')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser_args = argparse.ArgumentParser(description="Segment faces with uniface's XSeg and visualize the mask.")
    parser_args.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images')
    parser_args.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device for XSeg (PyTorch-backed)')
    parser_args.add_argument('--align_size', type=int, default=256, help='XSeg face alignment size')
    parser_args.add_argument('--blur_sigma', type=float, default=0, help='Gaussian blur sigma for mask smoothing (0 = raw)')
    parser_args.add_argument('--alpha', type=float, default=0.5, help='Overlay blend strength')
    parser_args.add_argument('--output_dir', type=str, default='output/face_segmentation', help='Directory to save visualizations and masks')
    args = parser_args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    detector = RetinaFace()
    xseg = XSeg(align_size=args.align_size, blur_sigma=args.blur_sigma, device=args.device)

    image_paths = gather_image_paths(args.input_path)
    if not image_paths:
        raise ValueError(f'No images found at {args.input_path}')

    num_skipped = 0
    for path in image_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path)
        if image is None:
            print(f'[skip] Failed to load image: {path}')
            num_skipped += 1
            continue

        mask, faces, num_segmented = segment_image(image, detector, xseg)
        if num_segmented == 0:
            print(f'[skip] No face with landmarks detected: {path} ({len(faces)} face(s) found)')
            num_skipped += 1
            continue

        overlay = apply_mask_overlay(image, mask, alpha=args.alpha)

        vis_path = os.path.join(args.output_dir, f'{name}_segmentation.png')
        save_visualization(image, mask, overlay, vis_path, f'{name} ({num_segmented} face(s))')

        mask_path = os.path.join(args.output_dir, f'{name}_mask.png')
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        print(f'[ok] {name}: segmented {num_segmented}/{len(faces)} face(s), mean mask coverage {mask.mean():.3f}')

    num_processed = len(image_paths) - num_skipped
    print(f'\nProcessed {num_processed}/{len(image_paths)} images. Skipped {num_skipped}.')
    print(f'Saved visualizations and masks to {args.output_dir}')


if __name__ == '__main__':
    main()
