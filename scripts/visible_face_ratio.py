import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import uniface before inserting the repo root onto sys.path below (needed for
# preprocessing.cropping) - the repo root contains a uniface/ checkout directory
# that would otherwise shadow the pip-installed uniface package of the same name.
from uniface.parsing import XSeg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cropping import build_retinaface_detector, get_crop_transform, warp_crop  # noqa: E402

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
BOX_COLOR = '#1c5cab'


def is_image_file(filepath):
    return os.path.splitext(filepath)[1].lower() in VALID_IMAGE_EXTENSIONS


def gather_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if is_image_file(f))
        return [os.path.join(input_path, f) for f in files]
    raise ValueError(f"input_path '{input_path}' is not a file or directory")


def detect_and_crop(image, face_detector, scale=1.4, image_size=224):
    """Detect a face once, then build the crop and map that same detection's
    tight box + 5-point landmarks into the crop's coordinate space.

    Returns (cropped, box_crop, landmarks_5pt_crop) or None if no face (with
    landmarks) was detected.
    """
    detections = face_detector(image, rgb=False)
    if detections is None:
        return None
    detections = np.asarray(detections)
    if detections.size == 0:
        return None
    detections = detections.reshape(-1, detections.shape[-1])
    det = detections[0]
    left, top, right, bottom = det[:4]
    if right <= left or bottom <= top or det.shape[0] < 15:
        return None

    tight_box = np.array([left, top, right, bottom], dtype=np.float32)
    tform = get_crop_transform(image, tight_box, scale=scale, image_size=image_size)
    cropped = warp_crop(image, tform, image_size=image_size)

    box_corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    box_corners_crop = tform(box_corners)
    box_crop = np.array([
        box_corners_crop[:, 0].min(), box_corners_crop[:, 1].min(),
        box_corners_crop[:, 0].max(), box_corners_crop[:, 1].max(),
    ], dtype=np.float32)

    landmarks_5pt_crop = tform(det[5:15].reshape(5, 2)).astype(np.float32)

    return cropped, box_crop, landmarks_5pt_crop


def compute_box_area(box):
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def compute_mask_area(mask, threshold):
    return float(np.count_nonzero(mask > threshold))


def apply_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.45):
    overlay = image.copy().astype(np.float32)
    color_overlay = np.zeros_like(image, dtype=np.float32)
    color_overlay[:] = color
    mask_3ch = mask[..., np.newaxis]
    overlay = overlay * (1 - mask_3ch * alpha) + color_overlay * mask_3ch * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def save_visualization(cropped_bgr, box, mask, result, out_path, title):
    rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    box_x = [box[0], box[2], box[2], box[0], box[0]]
    box_y = [box[1], box[1], box[3], box[3], box[1]]
    overlay_rgb = cv2.cvtColor(apply_mask_overlay(cropped_bgr, mask), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].fill(box_x, box_y, facecolor=BOX_COLOR, alpha=0.25, edgecolor=BOX_COLOR, linewidth=1.5)
    axes[0].set_title(f'Face box  (A = {result["box_area"]:.0f} px)')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title(f'Segmentation mask  (B = {result["mask_area"]:.0f} px)')
    axes[1].axis('off')

    axes[2].imshow(overlay_rgb)
    axes[2].plot(box_x, box_y, c=BOX_COLOR, linewidth=1.5)
    axes[2].set_title(f'Overlap  (B/A = {result["ratio_mask_to_box"]:.3f})')
    axes[2].axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser_args = argparse.ArgumentParser(
        description='Compute the visible face region metric: ratio of XSeg segmentation '
                     'mask area to the RetinaFace detection box area.')
    parser_args.add_argument('--input_path', type=str, required=True, help='Path to an image file or a directory of images')
    parser_args.add_argument('--device', type=str, default='cuda', help='Device for the RetinaFace detector and XSeg (PyTorch-backed)')
    parser_args.add_argument('--scale', type=float, default=1.4, help='Crop scale factor relative to the detected face box')
    parser_args.add_argument('--crop_size', type=int, default=224, help='Output crop size (square)')
    parser_args.add_argument('--mask_threshold', type=float, default=0.5, help='Threshold above which a mask pixel counts toward area B')
    parser_args.add_argument('--align_size', type=int, default=256, help='XSeg face alignment size')
    parser_args.add_argument('--blur_sigma', type=float, default=0, help='Gaussian blur sigma for XSeg mask smoothing (0 = raw)')
    parser_args.add_argument('--output_dir', type=str, default='output/visible_face_ratio', help='Directory to save visualizations and results')
    args = parser_args.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    face_detector = build_retinaface_detector(args.device)
    xseg = XSeg(align_size=args.align_size, blur_sigma=args.blur_sigma, device=args.device)

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

        detection = detect_and_crop(image, face_detector, scale=args.scale, image_size=args.crop_size)
        if detection is None:
            print(f'[skip] No face detected: {path}')
            results[name] = None
            num_skipped += 1
            continue
        cropped, box, landmarks_5pt = detection

        box_area = compute_box_area(box)

        mask = xseg.parse(cropped, landmarks=landmarks_5pt)
        mask_area = compute_mask_area(mask, args.mask_threshold)
        ratio = mask_area / box_area if box_area > 0 else float('nan')

        result = {'box_area': box_area, 'mask_area': mask_area, 'ratio_mask_to_box': float(ratio)}
        results[name] = result

        vis_path = os.path.join(args.output_dir, f'{name}_visible_region.png')
        save_visualization(cropped, box, mask, result, vis_path, name)

        print(f'[ok] {name}: A={box_area:.0f}  B={mask_area:.0f}  B/A={ratio:.3f}')

    results_path = os.path.join(args.output_dir, 'visible_face_ratio.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    num_processed = len(image_paths) - num_skipped
    print(f'\nProcessed {num_processed}/{len(image_paths)} images. Skipped {num_skipped}.')
    print(f'Saved visualizations and {results_path}')


if __name__ == '__main__':
    main()
