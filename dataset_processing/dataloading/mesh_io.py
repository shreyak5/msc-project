from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

EXPECTED_NUM_VERTICES = 5023


def load_flame_vertices(path: str | Path) -> torch.Tensor:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".ply":
        verts = _load_ply_vertices(path)
    elif ext == ".obj":
        verts = _load_obj_vertices(path)
    elif ext == ".json":
        verts = _load_json_vertices(path)
    else:
        raise ValueError(f"unsupported FLAME mesh extension: {ext} ({path})")

    if verts.shape != (EXPECTED_NUM_VERTICES, 3):
        raise ValueError(f"{path}: expected ({EXPECTED_NUM_VERTICES}, 3) vertices, got {tuple(verts.shape)}")
    return verts


def _load_ply_vertices(path: Path) -> torch.Tensor:
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"]
    coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    return torch.from_numpy(coords)


def _load_obj_vertices(path: Path) -> torch.Tensor:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return torch.tensor(verts, dtype=torch.float32)


def _load_json_vertices(path: Path) -> torch.Tensor:
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data["vertices"], dtype=torch.float32)