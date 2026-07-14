import numpy as np
import cv2
from skimage.transform import estimate_transform, warp


def build_retinaface_detector(device, threshold=0.8, model_name='mobilenet0.25'):
    from ibug.face_detection import RetinaFacePredictor
    return RetinaFacePredictor(
        threshold=threshold,
        device=device,
        model=RetinaFacePredictor.get_model(model_name))


def _detect_primary_face(image, face_detector):
    """Runs the detector once and returns its raw top detection row, or None if
    no face was found / the box is degenerate. RetinaFacePredictor's output is
    (N, 15) per detection: box(4) + score(1) + 5-point landmarks(10) - shared by
    get_face_box and get_face_landmarks so neither needs its own detector call."""
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

    return det


def get_face_box(image, face_detector):
    det = _detect_primary_face(image, face_detector)
    if det is None:
        return None
    return np.array(det[:4], dtype=np.float32)


def get_face_landmarks(image, face_detector):
    """5-point landmarks (left eye, right eye, nose tip, left mouth corner,
    right mouth corner - RetinaFace's own convention) from the same raw
    detection get_face_box uses, columns [5:15] reshaped to (5, 2). Used for
    MICA's own ArcFace-style alignment crop (dataset_processing/dataloading/
    mica_cache.py) - a separate, tighter, pose-normalizing alignment computed
    directly from these 5 points against a fixed destination template, not
    derived from crop_face's own loose box crop."""
    det = _detect_primary_face(image, face_detector)
    if det is None:
        return None
    return det[5:15].reshape(5, 2).astype(np.float32)


# Standard ArcFace/insightface 5-point destination template (left eye, right eye,
# nose tip, left mouth corner, right mouth corner) for a 112x112 canonical aligned
# crop - the same fixed constant SMIRK/insightface use (datasets/base_dataset.py's
# arcface_dst). Unlike SMIRK's estimate_norm, no 112/128-multiple generalization -
# this project only ever aligns to 112 (model/constants.py's MICA_IMAGE_SIZE).
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def get_arcface_transform(landmarks_5pt):
    return estimate_transform('similarity', landmarks_5pt, ARCFACE_DST)


def warp_arcface_crop(image, tform, image_size=112):
    return warp(image, tform.inverse, output_shape=(image_size, image_size), preserve_range=True).astype(np.uint8)


def crop_face_arcface(image, face_detector, image_size=112):
    """MICA/ArcFace-style pose-normalizing alignment crop (dataset_processing/
    dataloading/mica_cache.py) - a separate, tighter alignment from the original
    image than crop_face's loose box crop, computed directly from RetinaFace's
    5-point landmarks against the fixed ARCFACE_DST template, not derived from
    crop_face's own box/transform."""
    landmarks_5pt = get_face_landmarks(image, face_detector)
    if landmarks_5pt is None:
        return None, None

    tform = get_arcface_transform(landmarks_5pt)
    cropped_image = warp_arcface_crop(image, tform, image_size=image_size)
    return cropped_image, tform


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
