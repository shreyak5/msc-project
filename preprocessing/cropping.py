import numpy as np
import cv2
from skimage.transform import estimate_transform, warp


def build_retinaface_detector(device, threshold=0.8, model_name='mobilenet0.25'):
    from ibug.face_detection import RetinaFacePredictor
    return RetinaFacePredictor(
        threshold=threshold,
        device=device,
        model=RetinaFacePredictor.get_model(model_name))


def get_face_box(image, face_detector):
    detections = face_detector(image, rgb=False)
    if detections is None:
        return None

    detections = np.asarray(detections)
    if detections.size == 0:
        return None

    detections = detections.reshape(-1, detections.shape[-1])
    det = detections[0]
    left, top, right, bottom = det[:4]

    if right <= left or bottom <= top:
        return None

    return np.array([left, top, right, bottom], dtype=np.float32)


def get_crop_transform(frame, box, scale=1.4, image_size=224):
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(frame.shape[1] - 1, right)
    bottom = min(frame.shape[0] - 1, bottom)

    width = right - left
    height = bottom - top
    old_size = max(width, height)
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    size = int(old_size * scale)

    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform('similarity', src_pts, dst_pts)


def warp_crop(image, tform, image_size=224):
    return warp(image, tform.inverse, output_shape=(image_size, image_size), preserve_range=True).astype(np.uint8)


def crop_face(image, face_detector, scale=1.4, image_size=224):
    box = get_face_box(image, face_detector)
    if box is None:
        return None, None

    tform = get_crop_transform(image, box, scale=scale, image_size=image_size)
    cropped_image = warp_crop(image, tform, image_size=image_size)
    return cropped_image, tform
