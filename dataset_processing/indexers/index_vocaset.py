"""Index VOCASET into the common manifest format.

Raw layout: face_datasets/vocaset/FaceTalk_<date>_<id>_TA/sentenceNN/sentenceNN.NNNNNN.ply
(FLAME-topology mesh, 5023 verts, same 12 subjects as CoMA) paired with
face_datasets/vocaset/vocaset_images/imagessubject<N>/FaceTalk_.../sentenceNN/sentenceNN.NNNNNN.26_C.jpg
(N = 1-indexed position of the subject folder in sorted order; note the
"imagessubject" directory name has a double s, unlike CoMA's "imagesubject").
Camera 26_C is used exclusively, per project decision.

Verified across all 480 subject/sentence combinations that every mesh frame
has a matching 26_C image at the same index (0 mismatches out of 123710 mesh
frames), same as CoMA. Each row bundles a whole sentence via image_paths/
flame_mesh_paths lists (ordered, index-aligned), since VOCASET is a video
dataset (per project decision) even though frames ship as discrete files.
"""

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

            mesh_paths, image_paths = [], []
            for mesh_path in sorted(sent_dir.glob(f"{sentence}.*.ply")):
                m = re.match(rf"^{re.escape(sentence)}\.(\d+)\.ply$", mesh_path.name)
                if not m:
                    continue
                frame_idx_str = m.group(1)
                image_path = img_sent_dir / f"{sentence}.{frame_idx_str}.{CAMERA}.jpg"
                if not image_path.exists():
                    missing_images += 1
                    continue
                mesh_paths.append(str(mesh_path))
                image_paths.append(str(image_path))

            if not image_paths:
                continue

            yield ManifestRow(
                dataset="vocaset",
                sample_id=f"{subject}_{sentence}",
                subject_id=subject,
                dimensionality="3d",
                modality="video",
                image_paths=image_paths,
                flame_mesh_paths=mesh_paths,
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