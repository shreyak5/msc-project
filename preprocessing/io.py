import os

import cv2

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}


def is_image_file(filepath):
    return os.path.splitext(filepath)[1].lower() in VALID_IMAGE_EXTENSIONS


def load_image_sequence(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"Input path '{directory}' is not a directory when --image_seq is True")

    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    if not files:
        raise ValueError(f"No files found in directory '{directory}'")

    non_image_files = [f for f in files if not is_image_file(f)]
    if non_image_files:
        raise ValueError(f"Non-image files found in directory: {non_image_files}")

    images = []
    first_shape = None
    for filename in files:
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")

        if first_shape is None:
            first_shape = image.shape
        elif image.shape != first_shape:
            raise ValueError(f"Image '{filename}' has shape {image.shape}, but expected {first_shape}")

        images.append(image)

    return images


def load_frames(input_path, image_seq, fps):
    """Load frames from a video file or a directory of image frames.

    Returns (frames, video_fps, input_name).
    """
    if image_seq:
        frames = load_image_sequence(input_path)
        video_fps = fps
        input_name = os.path.basename(input_path.rstrip('/'))
        return frames, video_fps, input_name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Error opening video file: {input_path}')
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps
    input_name = os.path.splitext(os.path.basename(input_path))[0]

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, video_fps, input_name
