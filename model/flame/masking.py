"""Adapted from SMIRK (github.com/georgeretsi/smirk, MIT License, Copyright (c)
2024 George Retsinas)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import constants
from model.flame.flame import vertices2landmarks
from model.flame.renderer import face_vertices, vertex_normals

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_probabilities_per_flame_triangle(
    path: str | Path = _REPO_ROOT / constants.FLAME_MASKS_TRIANGLES_PATH,
) -> torch.Tensor:
    """path's file maps FLAME region name -> triangle indices in that region.
    Returns (NUM_FLAME_FACES,) per-triangle sampling weight (model.constants.
    FLAME_MASK_AREA_WEIGHTS), used to bias mesh_based_mask_uniform_faces's sampling."""
    flame_masks_triangles = np.load(path, allow_pickle=True).item()

    face_probabilities = torch.zeros(constants.NUM_FLAME_FACES)
    for area, weight in constants.FLAME_MASK_AREA_WEIGHTS.items():
        face_probabilities[flame_masks_triangles[area]] = weight
    return face_probabilities


def triangle_area(vertices: torch.Tensor) -> torch.Tensor:
    """vertices: (..., 3, 2) xy coordinates of a triangle's 3 corners -> (...,) area,
    via the shoelace formula."""
    x1, y1 = vertices[..., 0, 0], vertices[..., 0, 1]
    x2, y2 = vertices[..., 1, 0], vertices[..., 1, 1]
    x3, y3 = vertices[..., 2, 0], vertices[..., 2, 1]
    return 0.5 * torch.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)


def random_barycentric(num: int = 1) -> torch.Tensor:
    """Returns (num, 3) barycentric coordinates, uniformly sampled within a triangle
    (reflect-outside-the-triangle trick, since (u,v) sampled uniformly in the unit
    square only lands inside the triangle in the u+v<=1 half)."""
    u, v = torch.rand(num), torch.rand(num)
    outside_triangle = u + v > 1
    u[outside_triangle], v[outside_triangle] = 1 - u[outside_triangle], 1 - v[outside_triangle]

    alpha = 1 - (u + v)
    beta = u
    gamma = v
    return torch.stack((alpha, beta, gamma), dim=1)


def mesh_based_mask_uniform_faces(
    flame_trans_verts: torch.Tensor,
    flame_faces: torch.Tensor,
    face_probabilities: torch.Tensor,
    mask_ratio: float = constants.MASK_RATIO,
    coords: dict[str, torch.Tensor] | None = None,
    image_size: int = constants.RENDERER_IMAGE_SIZE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = flame_trans_verts.size(0)
    device = flame_trans_verts.device
    num_points_to_sample = int(mask_ratio * image_size * image_size)

    flame_faces_expanded = flame_faces.expand(batch_size, -1, -1)

    if coords is None:
        transformed_normals = vertex_normals(flame_trans_verts, flame_faces_expanded)
        transformed_face_normals = face_vertices(transformed_normals, flame_faces_expanded)
        transformed_face_normals = transformed_face_normals[:, :, :, 2].mean(dim=-1)
        face_probabilities = face_probabilities.repeat(batch_size, 1).to(device)

        # zero out probability for back-facing triangles
        face_probabilities = torch.where(
            transformed_face_normals < 0.05, face_probabilities, torch.zeros_like(transformed_face_normals)
        )

        # scale by each triangle's visible (xy-projected) area, so sampling density
        # is uniform over visible image area rather than uniform per-triangle (which
        # would oversample tiny triangles relative to large ones).
        fv = face_vertices(flame_trans_verts, flame_faces_expanded)
        xy_area = triangle_area(fv)
        face_probabilities = face_probabilities * xy_area

        # torch.multinomial requires every row to sum > 0 - a degenerate pose (all
        # triangles back-facing or zero projected area) otherwise crashes the CUDA
        # kernel and takes down the whole distributed job. Fall back to uniform
        # sampling for just the affected batch elements. NaN/Inf (e.g. from an
        # unsupervised/garbage camera scale multiplying a zeroed-out probability
        # into 0*inf=nan) must be checked separately from sum<=0: a NaN sum
        # compares False to <=0, so it would otherwise slip through this guard.
        degenerate = ~torch.isfinite(face_probabilities).all(dim=1) | (face_probabilities.sum(dim=1) <= 0)
        if degenerate.any():
            print(
                f"warning: degenerate face_probabilities for {degenerate.sum().item()} batch "
                "element(s), falling back to uniform sampling"
            )
            face_probabilities[degenerate] = 1.0

        sampled_faces_indices = torch.multinomial(face_probabilities, num_points_to_sample, replacement=True).to(
            device
        )
        barycentric_coords = random_barycentric(num=batch_size * num_points_to_sample).to(device)
        barycentric_coords = barycentric_coords.view(batch_size, num_points_to_sample, 3)
    else:
        sampled_faces_indices = coords["sampled_faces_indices"]
        barycentric_coords = coords["barycentric_coords"]

    npoints = vertices2landmarks(flame_trans_verts, flame_faces, sampled_faces_indices, barycentric_coords)

    npoints = 0.5 * (1 + npoints) * image_size
    npoints = npoints.long()
    npoints[..., 1] = torch.clamp(npoints[..., 1], 0, image_size - 1)
    npoints[..., 0] = torch.clamp(npoints[..., 0], 0, image_size - 1)

    return npoints, {"sampled_faces_indices": sampled_faces_indices, "barycentric_coords": barycentric_coords}


def transfer_pixels(
    img: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor, rbound: torch.Tensor | None = None
) -> torch.Tensor:
    """Builds an otherwise-all-zero image where each points2[b,i] pixel gets img's
    pixel value from points1[b,i]. When points1 == points2 (the reconstruction-pass
    use), this just extracts the sampled points' real values into a sparse image.
    When they differ (the cycle/augmentation-pass use), it transfers pixel *values*
    from the original mesh-surface locations to where those same mesh points project
    to *after* an expression augmentation has re-posed the mesh."""
    batch_size, channels, height, width = img.size()
    retained_pixels = torch.zeros_like(img)

    if rbound is not None:
        for b in range(batch_size):
            retained_pixels[b, :, points2[b, : rbound[b], 1], points2[b, : rbound[b], 0]] = img[
                b, :, points1[b, : rbound[b], 1], points1[b, : rbound[b], 0]
            ]
    else:
        retained_pixels[torch.arange(batch_size).unsqueeze(-1), :, points2[..., 1], points2[..., 0]] = img[
            torch.arange(batch_size).unsqueeze(-1), :, points1[..., 1], points1[..., 0]
        ]

    return retained_pixels


def masking(
    img: torch.Tensor,
    mask: torch.Tensor,
    extra_points: torch.Tensor,
    wr: int = constants.MASK_DILATION_RADIUS,
    rendered_mask: torch.Tensor | None = None,
    extra_noise: bool = True,
    random_mask: float = constants.MASK_RATIO,
) -> torch.Tensor:
    """img: (B,C,H,W) real photo, mask: (B,1,H,W) - 1 = kept as-is (background/scene
    context), 0 = candidate for full blackout (face interior); precomputed elsewhere,
    see module docstring, not computed here - extra_points: (B,C,H,W) sparse real-
    pixel-valued image from transfer_pixels. Returns the masked image fed into the
    UNet (Sec 2.5): background kept (minus an eroded safety margin, see wr), face
    interior fully blacked out except at extra_points' sparse locations."""
    batch_size, channels, height, width = img.size()

    # Erodes mask's value-1 region by wr pixels (equivalently dilates its value-0
    # region) - a safety margin, since `mask` is only a convex-hull-of-landmarks
    # approximation, not a pixel-perfect segmentation: without this margin, imprecision
    # right at the hull boundary could leave a thin sliver of real, unmasked face skin
    # immediately adjacent to the blacked-out region for the network to exploit.
    mask = 1 - F.max_pool2d(1 - mask, 2 * wr + 1, stride=1, padding=wr)

    if rendered_mask is not None:
        mask = mask * (1 - rendered_mask)

    masked_img = img * mask

    if extra_noise:
        noise_mult = torch.randn(extra_points.shape, device=img.device) * 0.05 + 1
        extra_points = extra_points * noise_mult

    if random_mask > 0:
        random_drop = torch.bernoulli(torch.ones((batch_size, 1, height, width)) * random_mask).to(img.device)
        random_drop = 1 - F.max_pool2d(random_drop, 11, stride=1, padding=5)
        extra_points = extra_points * random_drop

    masked_img[extra_points > 0] = extra_points[extra_points > 0]
    return masked_img.detach()
