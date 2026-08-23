import cv2

from preprocessing.cropping import build_retinaface_detector, get_face_box

INPUT_PATH = "samples/no-occlusion/000006.jpg"
OUTPUT_PATH = "output/retinaface_box_example.jpg"

image = cv2.imread(INPUT_PATH)

detector = build_retinaface_detector(device="cpu")
box = get_face_box(image, detector)
if box is None:
    raise RuntimeError(f"No face detected in {INPUT_PATH}")

left, top, right, bottom = [int(v) for v in box]
annotated = image.copy()
cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 2)

cv2.imwrite(OUTPUT_PATH, annotated)
print(f"Box: left={left}, top={top}, right={right}, bottom={bottom}")
print(f"Saved to {OUTPUT_PATH}")
