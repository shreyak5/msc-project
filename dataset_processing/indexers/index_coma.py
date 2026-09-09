from __future__ import annotations

import re
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_ROOT = FACE_DATASETS_ROOT / "COMA"
IMAGES_ROOT = RAW_ROOT / "coma_images"
CAMERA = "26_C"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "coma.jsonl"


def find_subject_dirs():
    return sorted(d for d in RAW_ROOT.iterdir() if d.is_dir() and d.name.startswith("FaceTalk_"))


def build_rows():
    missing_images = 0
    for i, subject_dir in enumerate(find_subject_dirs(), start=1):
        subject = subject_dir.name
        img_subject_dir = IMAGES_ROOT / f"imagesubject{i}" / subject

        for expr_dir in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
            expr = expr_dir.name
            img_expr_dir = img_subject_dir / expr

            for mesh_path in sorted(expr_dir.glob(f"{expr}.*.ply")):
                m = re.match(rf"^{re.escape(expr)}\.(\d+)\.ply$", mesh_path.name)
                if not m:
                    continue
                frame_idx_str = m.group(1)
                image_path = img_expr_dir / f"{expr}.{frame_idx_str}.{CAMERA}.jpg"
                if not image_path.exists():
                    missing_images += 1
                    continue

                yield ManifestRow(
                    dataset="coma",
                    sample_id=f"{subject}_{expr}_{frame_idx_str}",
                    subject_id=subject,
                    dimensionality="3d",
                    modality="image",
                    image_paths=[str(image_path)],
                    flame_mesh_paths=[str(mesh_path)],
                    frame_index=int(frame_idx_str),
                    sequence_id=expr,
                    camera_id=CAMERA,
                    labels={"expression": expr},
                )

    if missing_images:
        print(f"coma: warning -- {missing_images} mesh frames had no matching {CAMERA} image, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"coma: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()