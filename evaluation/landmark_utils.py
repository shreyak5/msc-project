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


def build_mediapipe_detector(model_asset_path):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1)
    return vision.FaceLandmarker.create_from_options(options)


def run_mediapipe(detector, image):
    import mediapipe as mp

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = detector.detect(mp_image)
    if len(detection_result.face_landmarks) == 0:
        return None

    face_landmarks = detection_result.face_landmarks[0]
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i, landmark in enumerate(face_landmarks):
        landmarks[i] = [landmark.x * mp_image.width, landmark.y * mp_image.height, landmark.z]

    return landmarks


def build_fan_predictor(device, model_name='2dfan4'):
    """
    FAN model options: 2dfan4, 2dfan2, 2dfan2alt
    ref: https://github.com/ibug-group/face_alignment/blob/master/ibug/face_alignment/fan/fan_predictor.py
    """
    from ibug.face_alignment import FANPredictor
    return FANPredictor(device=device, model=FANPredictor.get_model(model_name))


def run_fan(face_detector, fan_predictor, image):
    detected_faces = face_detector(image, rgb=False)
    if detected_faces is None or len(detected_faces) == 0:
        return None, None

    landmarks, scores = fan_predictor(image, detected_faces, rgb=False)
    if len(landmarks) == 0:
        return None, None

    return landmarks[0], scores[0]