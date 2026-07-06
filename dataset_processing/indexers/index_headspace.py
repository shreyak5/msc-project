"""Index Headspace/LYHM into the common manifest format.

FLAME-registered meshes come from the MICA-FLAME sub-package, not the raw
Headspace scans (which are high-res, non-FLAME topology):
  face_datasets/headspace/MICA-FLAME/LYHM/registrations/<subject_id>/<timestamp>.obj
    (FLAME-topology mesh, 5023 verts; 1211 subjects, exactly one scan each)

Images live in a separate sub-package entirely:
  face_datasets/headspace/headspacePngTka/subjects/<subject_id>/<timestamp>/{1A,1B,1C,...,5C}.png
    (5 physical cameras x 3 exposures each)

subject_id and timestamp are shared keys across both sub-packages (verified:
all 1211 registration subjects have a matching timestamp folder on the image
side, with both 1C.png and 2C.png present -- 0 missing in either case), so
matching is a direct join, no fuzzy/nearest-index logic needed.

Per project decision, camera view alternates by (sorted) subject index --
even index gets 1C, odd gets 2C -- rather than fixing one view for the whole
dataset. Each subject has exactly one scan, so this is an image (not video)
dataset with one row per subject.
"""

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