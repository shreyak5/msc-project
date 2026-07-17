"""Crop a face, extract FAN landmarks + per-landmark confidence scores, and
visualize the scores to study whether occlusions lower confidence.

Usage:
    python scripts/visualize_landmark_confidence.py --input_path <image_or_dir>
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cropping import build_retinaface_detector, crop_face, get_cropped_face_box  # noqa: E402
from preprocessing.io import is_image_file  # noqa: E402
from utils.landmark_utils import build_fan_predictor, run_fan  # noqa: E402

# Sequential single-hue ramp (light = low confidence -> dark = high confidence).
CONFIDENCE_CMAP = LinearSegmentedColormap.from_list('confidence_blue', [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
])


def gather_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if is_image_file(f))
        return [os.path.join(input_path, f) for f in files]
    raise ValueError(f"input_path '{input_path}' is not a file or directory")


def save_visualization(cropped_bgr, landmarks, scores, out_path, image_name):
    rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    scatter = ax.scatter(
        landmarks[:, 0], landmarks[:, 1], c=scores, cmap=CONFIDENCE_CMAP,
        vmin=0.0, vmax=1.0, s=25, edgecolors='white', linewidths=0.5)
    ax.set_title(f'{image_name}  (mean confidence: {scores.mean():.3f})')
    ax.axis('off')
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('FAN landmark confidence')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Crop faces with RetinaFace, extract FAN landmarks + confidence scores, '
                     'and visualize per-landmark confidence.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the detectors on')
    parser.add_argument('--scale', type=float, default=1.4, help='Crop scale factor relative to the detected face box')
    parser.add_argument('--crop_size', type=int, default=224, help='Output crop size (square)')
    parser.add_argument('--low_conf_threshold', type=float, default=0.3, help='Score threshold below which a landmark is counted as low-confidence')
    parser.add_argument('--output_dir', type=str, default='output/landmark_confidence', help='Directory to save visualizations and scores')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    face_detector = build_retinaface_detector(args.device)
    fan_predictor = build_fan_predictor(args.device)

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

        cropped, _tform = crop_face(image, face_detector, scale=args.scale, image_size=args.crop_size)
        if cropped is None:
            print(f'[skip] No face detected: {path}')
            results[name] = None
            num_skipped += 1
            continue

        fan_box = get_cropped_face_box(image_size=args.crop_size, scale=args.scale)
        landmarks, scores = run_fan(fan_predictor, cropped, fan_box)
        if landmarks is None:
            print(f'[skip] FAN failed to detect landmarks in crop: {path}')
            results[name] = None
            num_skipped += 1
            continue

        vis_path = os.path.join(args.output_dir, f'{name}_landmarks.png')
        save_visualization(cropped, landmarks, scores, vis_path, name)

        num_below_threshold = int(np.sum(scores < args.low_conf_threshold))
        results[name] = {
            'scores': scores.tolist(),
            'mean': float(scores.mean()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'num_below_threshold': num_below_threshold,
        }
        print(f'[ok] {name}: mean confidence {scores.mean():.3f} '
              f'({num_below_threshold} landmarks below {args.low_conf_threshold})')

    scores_path = os.path.join(args.output_dir, 'confidence_scores.json')
    with open(scores_path, 'w') as f:
        json.dump(results, f, indent=2)

    num_processed = len(image_paths) - num_skipped
    print(f'\nProcessed {num_processed}/{len(image_paths)} images. Skipped {num_skipped}.')
    print(f'Saved visualizations and {scores_path}')


if __name__ == '__main__':
    main()
