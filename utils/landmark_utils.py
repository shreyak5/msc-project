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
    import cv2
    import mediapipe as mp
    import numpy as np

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


def run_fan(fan_predictor, image, box):
    """box: (4,) [left, top, right, bottom] - the face's box within `image`. No
    face_detector here: callers operating on an already-cropped image (via
    preprocessing.cropping.crop_face) should pass preprocessing.cropping.
    get_cropped_face_box(image_size, scale)'s fixed, analytically-derived box
    rather than re-detecting - see that function's docstring for why re-
    detection would be redundant. FAN itself has no "no face" failure mode
    once given a box (unlike the old detector-based version) - it will always
    attempt to produce landmarks for whatever box it's given."""
    import numpy as np
    landmarks, scores = fan_predictor(image, np.asarray(box), rgb=False)
    if len(landmarks) == 0:
        return None, None
    return landmarks[0], scores[0]
