"""FLAME (Faces Learned with an Articulated Model and Expression) differentiable
mesh model (implementation-plan.md Sec 9: "Reuse from SMIRK repo: ... FLAME wrapper
(incl. eyelid blendshapes)").

Adapted from SMIRK (Retsinas et al., CVPR 2024, https://github.com/georgeretsi/smirk,
src/FLAME/FLAME.py, MIT License, Copyright (c) 2024 George Retsinas), itself adapted
from Sanyal et al.'s FLAME_PyTorch (MIT License for the wrapper class itself,
https://github.com/soubhiksanyal/FLAME_PyTorch) - see model/flame/lbs.py's docstring
for the underlying skinning math's own (non-commercial research) license chain.
SMIRK's own addition on top of FLAME_PyTorch: eyelid blendshapes (l_eyelid.npy/
r_eyelid.npy, copied into assets/, MIT licensed as part of SMIRK's contribution).

The FLAME *model* itself (assets/FLAME2020/*.pkl) is separately licensed by the Max
Planck Institute for Intelligent Systems for non-commercial research use only - see
assets/FLAME2020/Readme.pdf. Use here is non-commercial academic research.

Excludes FLAME.py's get_landmarks/_vertices2landmarks/seletec_3d68 methods: verified
(via grep) unused anywhere in SMIRK's own codebase, and containing genuine bugs
(undefined names) that happen to never be triggered because nothing calls them.
Also excludes:
- the zero_expression/zero_shape/zero_pose forward() flags and the expression/shape
  zero-padding-if-too-short logic: both existed only because SMIRK's own encoder has
  a configurable, possibly-smaller n_exp - our SViT/Heads always produce exactly
  n_exp/n_shape-sized tensors, and a caller wanting a "zeroed" FLAME evaluation can
  just pass zero tensors directly.
- landmarks_fan_3d (the fixed, non-dynamic-contour 68-point 3D landmark set) and its
  full_lmk_faces_idx/full_lmk_bary_coords buffers: unused by any loss in Sec 6 (the
  "Landmark" loss projects landmarks_fan/landmarks_mp to 2D and compares against
  detected 2D landmarks; the "Mesh" loss compares all vertices directly) - this was
  only ever loaded/computed in SMIRK's wrapper for generality, not something this
  project's losses consume.

Camera token layout (Sec 2.2, model/constants.py): [scale(1), global_rotation(3),
translation_xy(2)]. Only global_rotation is consumed here - scale/translation feed
the renderer's weak-perspective projection (model/renderer.py), not FLAME itself.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import numpy as np

# Compatibility shims so the old chumpy-pickled FLAME model can be unpickled under
# modern numpy/python: unpickling assets/FLAME2020/generic_model.pkl transitively
# imports chumpy (some fields are stored as chumpy Ch objects), whose old code uses
# an inspect API and numpy type aliases that were both since removed upstream (same
# shims as scripts/compare_flame_models.py, matching SMIRK's own FLAME.py which
# applies the numpy aliases the same way).
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _val in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
                     ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import torch
import torch.nn as nn

from model import constants
from model.flame.lbs import batch_rodrigues, lbs, rot_mat_to_euler, vertices2landmarks

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NECK_JOINT_IDX = 1


def _to_tensor(array, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(array, dtype=dtype)


def _to_np(array, dtype=np.float32) -> np.ndarray:
    """Plain ndarray, scipy sparse matrix, or chumpy Ch object -> plain ndarray."""
    if hasattr(array, "toarray"):  # scipy sparse
        array = array.toarray()
    elif hasattr(array, "r"):  # chumpy Ch
        array = np.asarray(array.r)
    return np.array(array, dtype=dtype)


class _Struct:
    """Turns a dict (the unpickled FLAME model) into attribute access."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """Given FLAME parameters, generates a differentiable mesh + 2D/MediaPipe
    landmarks. n_shape/n_exp default to this project's dims (Sec 2.2) - TokenFace/
    this plan uses 100 expression coefficients, not SMIRK's own default of 50."""

    NUM_VERTICES = constants.EXPECTED_NUM_FLAME_VERTICES

    def __init__(
        self,
        n_shape: int = constants.FLAME_SHAPE_DIM,
        n_exp: int = constants.FLAME_EXPRESSION_DIM,
        flame_model_path: str | Path = _REPO_ROOT / constants.FLAME_MODEL_PATH,
        flame_lmk_embedding_path: str | Path = _REPO_ROOT / constants.FLAME_LMK_EMBEDDING_PATH,
        mediapipe_lmk_embedding_path: str | Path = _REPO_ROOT / constants.FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH,
        l_eyelid_path: str | Path = _REPO_ROOT / constants.FLAME_L_EYELID_PATH,
        r_eyelid_path: str | Path = _REPO_ROOT / constants.FLAME_R_EYELID_PATH,
    ):
        super().__init__()
        self.n_shape = n_shape
        self.n_exp = n_exp

        with open(flame_model_path, "rb") as f:
            flame_model = _Struct(**pickle.load(f, encoding="latin1"))

        self.register_buffer("faces_tensor", _to_tensor(_to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer("v_template", _to_tensor(_to_np(flame_model.v_template)))

        # flame_model.shapedirs is (5023, 3, 400): columns 0:300 are the identity/shape
        # PCA basis, 300:400 are the expression basis (fixed by how FLAME's authors
        # built the model file). Only slice out a subset when asking for fewer than the
        # full complement - at our actual n_shape=300/n_exp=100 (all of both), slicing
        # and re-concatenating would just reconstruct the identical (5023,3,400) tensor.
        shapedirs = _to_tensor(_to_np(flame_model.shapedirs))
        if n_shape < 300 or n_exp < 100:
            shapedirs = torch.cat([shapedirs[:, :, :n_shape], shapedirs[:, :, 300 : 300 + n_exp]], 2)
        self.register_buffer("shapedirs", shapedirs)

        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", _to_tensor(_to_np(posedirs)))

        self.register_buffer("J_regressor", _to_tensor(_to_np(flame_model.J_regressor)))
        parents = _to_tensor(_to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", _to_tensor(_to_np(flame_model.weights)))

        self.register_buffer("l_eyelid", torch.from_numpy(np.load(l_eyelid_path)).float()[None])
        self.register_buffer("r_eyelid", torch.from_numpy(np.load(r_eyelid_path)).float()[None])

        # FLAME's eyeball/neck joints are fixed (not predicted) - matching this
        # design's absence of any eye/neck component token, same as SMIRK's own choice.
        self.register_buffer("eye_pose", torch.zeros(1, 6))
        self.register_buffer("neck_pose", torch.zeros(1, 3))

        # Static landmarks: fixed mesh triangles/barycentric-coords, same regardless of
        # head pose (eye corners, nose tip, mouth corners, etc.). Dynamic landmarks: a
        # precomputed lookup table (indexed by rounded head-yaw-angle) for the jaw/face
        # contour points, whose visible mesh location shifts as the head turns.
        lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")[()]
        self.register_buffer("lmk_faces_idx", torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"]).long())
        self.register_buffer("lmk_bary_coords", torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).float())
        self.register_buffer("dynamic_lmk_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long())
        self.register_buffer("dynamic_lmk_bary_coords", lmk_embeddings["dynamic_lmk_bary_coords"].float())

        # FLAME's joint hierarchy is root(global) -> neck -> jaw -> {leye, reye}. Each
        # joint's rotation is relative to its parent, so computing the neck's true
        # (absolute) rotation - needed to look up the dynamic contour landmarks above -
        # requires composing rotations along this chain, from the neck up to the root.
        neck_kin_chain = []
        curr_idx = torch.tensor(_NECK_JOINT_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        mp_lmk_embeddings = np.load(mediapipe_lmk_embedding_path)
        self.register_buffer(
            "mp_lmk_faces_idx", torch.from_numpy(mp_lmk_embeddings["lmk_face_idx"].astype("int32")).long()
        )
        self.register_buffer("mp_lmk_bary_coords", torch.from_numpy(mp_lmk_embeddings["lmk_b_coords"]).float())

    def _find_dynamic_lmk_idx_and_bcoords(self, full_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Selects the face-contour landmarks depending on the head's relative
        rotation (a fixed lookup table indexed by rounded yaw angle)."""
        batch_size = full_pose.shape[0]
        aa_pose = torch.index_select(full_pose.view(batch_size, -1, 3), 1, self.neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)

        rel_rot_mat = (
            torch.eye(3, device=full_pose.device, dtype=full_pose.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        )
        for idx in range(len(self.neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).long()
        neg_mask = y_rot_angle.lt(0).long()
        mask = y_rot_angle.lt(-39).long()
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(self.dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_bary_coords = torch.index_select(self.dynamic_lmk_bary_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_bary_coords

    def forward(
        self,
        shape_params: torch.Tensor,
        expression_params: torch.Tensor,
        jaw_params: torch.Tensor,
        eyelid_params: torch.Tensor,
        global_rotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """shape_params: (B, n_shape), expression_params: (B, n_exp), jaw_params:
        (B, 3), eyelid_params: (B, 2), global_rotation: (B, 3) - the latter two are
        the ComponentHeads' "expression" token split at NUM_EYELID_PARAMS and the
        "camera" token sliced at CAMERA_ROTATION_SLICE, respectively.

        Returns: vertices (B,5023,3), landmarks_fan (B,68,3) - pose-adjusted 2D/FAN-
        convention landmarks, meant to be projected to 2D and compared against
        detected 2D landmarks (Sec 6) - and landmarks_mp (B,105,3), same idea for
        FLAME's curated MediaPipe-convention landmark subset."""
        batch_size = shape_params.shape[0]

        eye_pose_params = self.eye_pose.expand(batch_size, -1)
        neck_pose_params = self.neck_pose.expand(batch_size, -1)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([global_rotation, neck_pose_params, jaw_params, eye_pose_params], dim=1)

        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        vertices = vertices + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
        vertices = vertices + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(full_pose)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks_fan = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        landmarks_mp = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.mp_lmk_faces_idx.repeat(batch_size, 1),
            self.mp_lmk_bary_coords.repeat(batch_size, 1, 1),
        )

        return {
            "vertices": vertices,
            "landmarks_fan": landmarks_fan,
            "landmarks_mp": landmarks_mp,
        }
