import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cropping import build_retinaface_detector, crop_face, get_cropped_face_box  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.landmark_utils import build_mediapipe_detector, build_fan_predictor, run_mediapipe, run_fan  # noqa: E402

# TODO - point this to an assets folder in project root
DEFAULT_MEDIAPIPE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'baselines', 'smirk_experiments', 'assets', 'face_landmarker.task')


def process_frame(frame, face_detector, mediapipe_detector, fan_predictor, scale, crop_size):
    cropped, tform = crop_face(frame, face_detector, scale=scale, image_size=crop_size)

    # Number of mediapipe landmarks = 478, fan landmarks = 68
    if cropped is None:
        mp_lmks = np.full((478, 3), np.nan, dtype=np.float32)
        fan_lmks = np.full((68, 2), np.nan, dtype=np.float32)
        # fan_scores = np.full((68,), np.nan, dtype=np.float32)
        return None, mp_lmks, fan_lmks

    mp_lmks = run_mediapipe(mediapipe_detector, cropped)
    if mp_lmks is None:
        mp_lmks = np.full((478, 3), np.nan, dtype=np.float32)

    # Fixed given scale/crop_size (see get_cropped_face_box's docstring) - no second,
    # redundant RetinaFace call on the already-cropped image just to get FAN a box.
    fan_box = get_cropped_face_box(image_size=crop_size, scale=scale)
    fan_lmks, _fan_scores = run_fan(fan_predictor, cropped, fan_box)
    if fan_lmks is None:
        fan_lmks = np.full((68, 2), np.nan, dtype=np.float32)
        # fan_scores = np.full((68,), np.nan, dtype=np.float32)

    return cropped, mp_lmks, fan_lmks


def draw_landmarks(image, mp_lmks, fan_lmks):
    vis = image.copy()
    if not np.isnan(mp_lmks).any():
        for x, y, _ in mp_lmks:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    if not np.isnan(fan_lmks).any():
        for x, y in fan_lmks:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
    return vis


def main():
    parser = argparse.ArgumentParser(description='Crop faces with RetinaFace and extract MediaPipe + FAN landmarks per frame.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to a video file or a directory of frames')
    parser.add_argument('--image_seq', action='store_true', help='Treat input_path as a directory of image frames')
    parser.add_argument('--fps', type=int, default=30, help='FPS for the input image sequence (only used with --image_seq) and for --vis_path output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the detectors on')
    parser.add_argument('--scale', type=float, default=1.4, help='Crop scale factor relative to the detected face box')
    parser.add_argument('--crop_size', type=int, default=224, help='Output crop size (square)')
    parser.add_argument('--mediapipe_model_path', type=str, default=DEFAULT_MEDIAPIPE_MODEL_PATH, help='Path to the MediaPipe face_landmarker.task asset')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the extracted landmarks')
    parser.add_argument('--vis_path', type=str, default=None, help='Visualisation path: Optional path to write an mp4 of cropped frames with landmarks drawn on them (e.g. /path/video_name.mp4)')
    args = parser.parse_args()

    face_detector = build_retinaface_detector(args.device)
    mediapipe_detector = build_mediapipe_detector(args.mediapipe_model_path)
    fan_predictor = build_fan_predictor(args.device)

    frames, video_fps, input_name = load_frames(args.input_path, args.image_seq, args.fps)

    vis_writer = None
    if args.vis_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.vis_path)), exist_ok=True)
        vis_writer = cv2.VideoWriter(
            args.vis_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (args.crop_size, args.crop_size))
        if not vis_writer.isOpened():
            raise RuntimeError(f'Failed to open VideoWriter for --vis_path: {args.vis_path}')
        print(f'Saving visualisation video to {os.path.abspath(args.vis_path)}')

    mp_landmarks_list = []
    fan_landmarks_list = []
    # fan_scores_list = []

    for frame in frames:
        cropped, mp_lmks, fan_lmks = process_frame(
            frame, face_detector, mediapipe_detector, fan_predictor, args.scale, args.crop_size)

        mp_landmarks_list.append(mp_lmks)
        fan_landmarks_list.append(fan_lmks)

        # fan_scores_list.append(fan_scores)

        if vis_writer is not None:
            vis_frame = cropped if cropped is not None else np.zeros((args.crop_size, args.crop_size, 3), dtype=np.uint8)
            vis_writer.write(draw_landmarks(vis_frame, mp_lmks, fan_lmks))

    if vis_writer is not None:
        vis_writer.release()

    out_dir = os.path.join(args.output_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'mediapipe_landmarks.npy'), np.stack(mp_landmarks_list))
    np.save(os.path.join(out_dir, 'fan_landmarks.npy'), np.stack(fan_landmarks_list))
    # np.save(os.path.join(out_dir, 'fan_scores.npy'), np.stack(fan_scores_list))

    print(f'Processed {len(frames)} frames. Saved landmarks to {out_dir}')


if __name__ == '__main__':
    main()