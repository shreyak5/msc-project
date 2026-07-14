"""Linear Blend Skinning (LBS) math underlying the FLAME model (model/flame.py).

This is standard, well-tested SMPL-family skinning math, reproduced as-is rather
than modified (implementation-plan.md: "Do not reimplement SMIRK components from
scratch"). Traced attribution chain: originally from the smplx package (Choutas et
al., https://github.com/vchoutas/smplx, smplx/lbs.py) - licensed for non-commercial
scientific research purposes only, proprietary to the Max Planck Institute for
Intelligent Systems (Max-Planck-Gesellschaft). FLAME_PyTorch (Sanyal et al.,
https://github.com/soubhiksanyal/FLAME_PyTorch) imports these functions directly
from the smplx package rather than bundling a copy. SMIRK (src/FLAME/lbs.py)
inlined a local copy of these same functions (to drop the smplx dependency) plus
its own eyelid-blendshape addition elsewhere; this file is adapted from SMIRK's copy.
Use here is non-commercial academic research, consistent with this license.

Excludes the module-level find_dynamic_lmk_idx_and_bcoords: verified (via grep)
unused anywhere in SMIRK's own codebase - FLAME.py defines and calls its own
method version instead (which doesn't need the unused `vertices` parameter this
one takes), so this module-level copy is dead code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rot_mat_to_euler(rot_mats: torch.Tensor) -> torch.Tensor:
    """Calculates rotation matrix to euler angles. Careful for extreme cases of
    euler angles like [0.0, pi, 0.0]."""
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def vertices2landmarks(
    vertices: torch.Tensor, faces: torch.Tensor, lmk_faces_idx: torch.Tensor, lmk_bary_coords: torch.Tensor
) -> torch.Tensor:
    """Calculates landmarks by barycentric interpolation.

    vertices: BxVx3, faces: Fx3 (long), lmk_faces_idx: L (long, or BxL),
    lmk_bary_coords: Lx3 (or BxLx3) -> landmarks: BxLx3.
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)
    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
    landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
    return landmarks


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """J_regressor: JxV, vertices: BxVx3 -> BxJx3 joint locations."""
    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    """betas: Bx(num_betas), shape_disps: Vx3x(num_betas) -> BxVx3 per-vertex
    displacement: Displacement[b,m,k] = sum_l betas[b,l] * shape_disps[m,k,l]."""
    return torch.einsum("bl,mkl->bmk", [betas, shape_disps])


def batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """rot_vecs: Nx3 axis-angle vectors -> Nx3x3 rotation matrices."""
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """R: Bx3x3 rotation matrices, t: Bx3x1 translation vectors -> Bx4x4 transforms."""
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """rot_mats: BxNx3x3, joints: BxNx3, parents: N (kinematic tree) -> (posed_joints
    BxNx3, rel_transforms BxNx4x4 relative to the root joint)."""
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(
        -1, joints.shape[1], 4, 4
    )

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose - no need for rotation, since
        # it's identity when at rest.
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs Linear Blend Skinning with the given shape and pose parameters.

    betas: BxNB shape parameters, pose: Bx(J+1)*3 axis-angle pose parameters,
    v_template: BxVx3 template mesh, shapedirs: Vx3xNB PCA shape displacements,
    posedirs: Px(V*3) pose PCA coefficients, J_regressor: JxV, parents: J
    (kinematic tree), lbs_weights: VxJ+1 skinning weights.
    Returns (verts BxVx3, joints BxJx3).
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed
