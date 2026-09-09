from __future__ import annotations

from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_ROOT = FACE_DATASETS_ROOT / "famos"
REG_ROOT = RAW_ROOT / "registrations"
IMG_ROOT = RAW_ROOT / "famos_images" / "downsampled_images_4"
CAMERA = "26_C"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "famos.jsonl"


def build_rows():
    missing_mesh = 0
    for subject_dir in sorted(d for d in IMG_ROOT.iterdir() if d.is_dir()):
        subject = subject_dir.name
        for seq_dir in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
            sequence = seq_dir.name
            reg_seq_dir = REG_ROOT / subject / sequence

            for frame_dir in sorted(d for d in seq_dir.iterdir() if d.is_dir()):
                frame_idx_str = frame_dir.name
                image_path = frame_dir / f"{sequence}.{frame_idx_str}.{CAMERA}.png"
                if not image_path.exists():
                    continue
                mesh_path = reg_seq_dir / f"{sequence}.{frame_idx_str}.ply"
                if not mesh_path.exists():
                    missing_mesh += 1
                    continue

                yield ManifestRow(
                    dataset="famos",
                    sample_id=f"{subject}_{sequence}_{frame_idx_str}",
                    subject_id=subject,
                    dimensionality="3d",
                    modality="image",
                    image_paths=[str(image_path)],
                    flame_mesh_paths=[str(mesh_path)],
                    frame_index=int(frame_idx_str),
                    sequence_id=sequence,
                    camera_id=CAMERA,
                    labels={"sequence": sequence},
                )

    if missing_mesh:
        print(f"famos: warning -- {missing_mesh} {CAMERA} image frames had no matching mesh, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"famos: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()