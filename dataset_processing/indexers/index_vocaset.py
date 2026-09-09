from __future__ import annotations

import re
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_ROOT = FACE_DATASETS_ROOT / "vocaset"
IMAGES_ROOT = RAW_ROOT / "vocaset_images"
CAMERA = "26_C"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "vocaset.jsonl"


def find_subject_dirs():
    return sorted(d for d in RAW_ROOT.iterdir() if d.is_dir() and d.name.startswith("FaceTalk_"))


def build_rows():
    missing_images = 0
    for i, subject_dir in enumerate(find_subject_dirs(), start=1):
        subject = subject_dir.name
        img_subject_dir = IMAGES_ROOT / f"imagessubject{i}" / subject

        for sent_dir in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
            sentence = sent_dir.name
            img_sent_dir = img_subject_dir / sentence

            for mesh_path in sorted(sent_dir.glob(f"{sentence}.*.ply")):
                m = re.match(rf"^{re.escape(sentence)}\.(\d+)\.ply$", mesh_path.name)
                if not m:
                    continue
                frame_idx_str = m.group(1)
                image_path = img_sent_dir / f"{sentence}.{frame_idx_str}.{CAMERA}.jpg"
                if not image_path.exists():
                    missing_images += 1
                    continue

                yield ManifestRow(
                    dataset="vocaset",
                    sample_id=f"{subject}_{sentence}_{frame_idx_str}",
                    subject_id=subject,
                    dimensionality="3d",
                    modality="image",
                    image_paths=[str(image_path)],
                    flame_mesh_paths=[str(mesh_path)],
                    frame_index=int(frame_idx_str),
                    sequence_id=sentence,
                    camera_id=CAMERA,
                    labels={"sentence": sentence},
                )

    if missing_images:
        print(f"vocaset: warning -- {missing_images} mesh frames had no matching {CAMERA} image, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"vocaset: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()