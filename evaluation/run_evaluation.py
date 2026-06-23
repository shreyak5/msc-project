import argparse
import json
import os
import sys

import cv2
import numpy as np

from eval_core import build_evaluators, evaluate_clip, METHOD_REGISTRY
from metrics import summarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.io import load_frames  # noqa: E402

DEFAULT_MEDIAPIPE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'baselines', 'smirk_experiments', 'assets', 'face_landmarker.task')


def make_vis_writer(path, fps, crop_size):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_size, crop_size))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter for: {path}')
    return writer


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a 3D reconstruction method against FAN/MediaPipe 2D landmarks on a video or image sequence.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to a video file or a directory of frames')
    parser.add_argument('--image_seq', action='store_true', help='Treat input_path as a directory of image frames')
    parser.add_argument('--fps', type=int, default=30, help='FPS for the input image sequence and for vis videos')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run detectors/method on')
    parser.add_argument('--crop_scale', type=float, default=1.4, help='Crop scale factor relative to the detected face box')
    parser.add_argument('--crop_size', type=int, default=224, help='Output crop size (square)')
    parser.add_argument('--mediapipe_model_path', type=str, default=DEFAULT_MEDIAPIPE_MODEL_PATH,
                         help='Path to the MediaPipe face_landmarker.task asset')
    parser.add_argument('--method', type=str, default='smirk', choices=sorted(METHOD_REGISTRY.keys()))
    parser.add_argument('--landmark_sets', type=str, nargs='+', default=['fan', 'mediapipe'],
                         choices=['fan', 'mediapipe'], help='Which landmark sets to evaluate')
    parser.add_argument('--fan_vis_path', type=str, default=None, help='Optional path to write a FAN overlay video')
    parser.add_argument('--mediapipe_vis_path', type=str, default=None,
                         help='Optional path to write a MediaPipe overlay video')
    parser.add_argument('--output_dir', type=str, default='evaluation/output', help='Directory to save error arrays/summary')
    args = parser.parse_args()

    enabled_sets = set(args.landmark_sets)
    vis_paths = {'fan': args.fan_vis_path, 'mediapipe': args.mediapipe_vis_path}

    evaluators = build_evaluators(args.method, args.device, args.crop_size, args.mediapipe_model_path, enabled_sets)

    frames, video_fps, input_name = load_frames(args.input_path, args.image_seq, args.fps)

    vis_writers = {}
    for name in enabled_sets:
        if vis_paths[name]:
            vis_writers[name] = make_vis_writer(vis_paths[name], video_fps, args.crop_size)
            print(f'Saving {name} visualisation video to {os.path.abspath(vis_paths[name])}')

    errors = evaluate_clip(frames, args.crop_scale, args.crop_size, evaluators, enabled_sets, vis_writers)

    for writer in vis_writers.values():
        writer.release()

    out_dir = os.path.join(args.output_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for name in enabled_sets:
        np.save(os.path.join(out_dir, f'{name}_errors.npy'), errors[name])
        summary[name] = summarize(errors[name])

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Processed {len(frames)} frames. Saved results to {out_dir}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
