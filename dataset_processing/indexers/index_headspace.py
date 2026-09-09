from __future__ import annotations

from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_ROOT = FACE_DATASETS_ROOT / "headspace"
REG_ROOT = RAW_ROOT / "MICA-FLAME" / "LYHM" / "registrations"
IMG_ROOT = RAW_ROOT / "headspacePngTka" / "subjects"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "headspace.jsonl"


def build_rows():
    subject_dirs = sorted(d for d in REG_ROOT.iterdir() if d.is_dir())
    missing = 0

    for i, subject_dir in enumerate(subject_dirs):
        subject = subject_dir.name
        mesh_files = list(subject_dir.glob("*.obj"))
        if len(mesh_files) != 1:
            missing += 1
            continue
        mesh_path = mesh_files[0]
        timestamp = mesh_path.stem

        view = "1C" if i % 2 == 0 else "2C"
        image_path = IMG_ROOT / subject / timestamp / f"{view}.png"
        if not image_path.exists():
            missing += 1
            continue

        yield ManifestRow(
            dataset="headspace",
            sample_id=f"{subject}_{timestamp}",
            subject_id=subject,
            dimensionality="3d",
            modality="image",
            image_paths=[str(image_path)],
            flame_mesh_paths=[str(mesh_path)],
            camera_id=view,
            labels={"timestamp": timestamp},
        )

    if missing:
        print(f"headspace: warning -- {missing} subjects had no usable mesh/image pair, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"headspace: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()