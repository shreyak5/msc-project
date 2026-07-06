"""Index FaMoS into the common manifest format.

Raw layout:
  face_datasets/famos/registrations/FaMoS_subject_NNN/<sequence>/<sequence>.NNNNNN.ply
    (FLAME-topology mesh, 5023 verts; dense/continuous frame numbering; 95 subjects)
  face_datasets/famos/famos_images/downsampled_images_4/FaMoS_subject_NNN/<sequence>/<frame>/<sequence>.<frame>.<camID>_<variant>.png
    (only 78 of the 95 subjects have images; only a sparse subset of frame
    indices were sampled per sequence)

Camera 26, variant C ("26_C") is used exclusively, per project decision.
Unlike CoMA/VOCASET, image coverage here is sparse and non-uniform rather
than dense/continuous, so FaMoS is indexed as an image dataset (one row per
frame) rather than bundling a sequence into one row. Verified that every
sampled 26_C image frame has a matching mesh at the exact same frame index
(0 missing out of 89596). Subjects 079-095 have no images at all and are
therefore absent from this manifest (mesh-only, not usable for this
image+mesh pairing).
"""

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