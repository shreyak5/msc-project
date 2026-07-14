"""Differentiable mesh rasterizer/renderer (implementation-plan.md Sec 9: "Reuse from
SMIRK repo: rasterizer ..."). Produces the grayscale-shaded mesh render fed into the
UNet (Sec 2.5) and projects 3D landmarks to 2D for the landmark loss (Sec 6).

Adapted from SMIRK (Retsinas et al., CVPR 2024, https://github.com/georgeretsi/smirk,
src/renderer/renderer.py + src/renderer/util.py, MIT License, Copyright (c) 2024
George Retsinas). face_vertices is itself borrowed by SMIRK from
daniilidis-group/neural_renderer (MIT License, Copyright (c) 2017 Hiroharu Kato,
2018 Nikos Kolotouros).

Uses pytorch3d (Meshes, rasterize_meshes) for the actual rasterization - a heavy
CUDA-compiled dependency the plan doesn't otherwise mention, but required to reuse
SMIRK's rasterizer directly rather than reimplementing it.

Does not load a separate head_template.obj mesh asset (unlike SMIRK's own renderer):
verified empirically that its face connectivity is identical to FLAME's own
faces_tensor (model/flame/flame.py), so faces are passed in directly instead of
loading a redundant duplicate asset. FLAME_masks.pkl (a curated vertex-region
annotation - face/neck/ears/scalp/...) is still a genuinely separate asset, used to
restrict rendering to the face region only (Sec 2.5/9, matching SMIRK's own
render_full_head=False default).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes

from model import constants

_REPO_ROOT = Path(__file__).resolve().parents[2]


def face_vertices(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """vertices: (B, V, 3), faces: (B, F, 3) vertex-index triples -> (B, F, 3, 3):
    for each face, the 3D positions of its 3 corner vertices."""
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(batch_size, dtype=torch.int32, device=device) * num_verts)[:, None, None]
    vertices = vertices.reshape((batch_size * num_verts, 3))
    return vertices[faces.long()]


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """vertices: (B, V, 3), faces: (B, F, 3) -> (B, V, 3) smoothed per-vertex unit
    normals: each face contributes an (unnormalized, area-weighted) normal to its 3
    corner vertices via cross products of edge vectors; contributions are summed per
    vertex (a vertex is shared by multiple faces) and normalized to unit length."""
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    normals = torch.zeros(batch_size * num_verts, 3, device=device)

    faces = faces + (torch.arange(batch_size, dtype=torch.int32, device=device) * num_verts)[:, None, None]
    vertices_faces = vertices.reshape((batch_size * num_verts, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1], dim=-1),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2], dim=-1),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0], dim=-1),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    return normals.reshape((batch_size, num_verts, 3))


def batch_orth_proj(points: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
    """Weak-perspective/orthographic projection. points: (B, N, 3), camera:
    (B, 3) = [scale, tx, ty] (model.constants.CAMERA_SCALE_SLICE/TRANSLATION_SLICE)
    -> (B, N, 3). Note: translates by the raw (unscaled) tx,ty, then scales the
    whole result - i.e. scale*(X + t), not the textbook scale*X + t. Preserved
    exactly as SMIRK has it (not a numerical bug: the camera-regressing network is
    trained end-to-end against whatever formula this is, so it learns tx/ty values
    correct for this specific parameterization - self-consistent either way)."""
    camera = camera.clone().view(-1, 1, 3)
    xy_translated = points[:, :, :2] + camera[:, :, 1:]
    translated = torch.cat([xy_translated, points[:, :, 2:]], dim=2)
    return camera[:, :, 0:1] * translated


def _keep_vertices_and_update_faces(faces: torch.Tensor, vertices_to_keep) -> torch.Tensor:
    """faces: (F, 3), vertices_to_keep: vertex indices to keep -> faces re-indexed
    to only that vertex subset, dropping any face that referenced a removed vertex."""
    device = faces.device
    if isinstance(vertices_to_keep, (list, np.ndarray)):
        vertices_to_keep = torch.tensor(vertices_to_keep, dtype=torch.long)
    vertices_to_keep = torch.unique(vertices_to_keep).to(device)

    max_vertex_index = faces.max().long().item() + 1
    mask = torch.zeros(max_vertex_index, dtype=torch.bool, device=device)
    mask[vertices_to_keep] = True

    new_vertex_indices = torch.full((max_vertex_index,), -1, dtype=torch.long, device=device)
    new_vertex_indices[mask] = torch.arange(len(vertices_to_keep), device=device)

    valid_faces_mask = (new_vertex_indices[faces] != -1).all(dim=1)
    return new_vertex_indices[faces[valid_faces_mask]]


class Renderer(nn.Module):
    """Rasterizes a FLAME mesh into a grayscale-shaded image (flat albedo x
    directional-light shading, Sec 2.5) and projects 3D points (mesh vertices,
    landmarks) to 2D via the weak-perspective camera (Sec 2.2)."""

    def __init__(
        self,
        faces: torch.Tensor,
        render_full_head: bool = constants.RENDERER_FULL_HEAD,
        image_size: int = constants.RENDERER_IMAGE_SIZE,
        flame_masks_path: str | Path = _REPO_ROOT / constants.RENDERER_FLAME_MASKS_PATH,
    ):
        """faces: (F, 3) long tensor, FLAME's face connectivity (e.g. FLAME.faces_tensor)."""
        super().__init__()
        self.image_size = image_size
        self.render_full_head = render_full_head

        # Build everything on CPU regardless of the input tensor's device - a caller's
        # module (e.g. FLAME) is often already .to(device)'d before its faces_tensor is
        # passed in here. register_buffer + the caller's own subsequent .to(device) on
        # this whole module moves everything (including actual rendering at train/
        # inference time) to GPU together afterward - this setup arithmetic itself is
        # one-time, cheap index bookkeeping, not anything performance-sensitive.
        faces = faces.detach().cpu().long().unsqueeze(0)  # (1, F, 3)
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.0

        with open(flame_masks_path, "rb") as f:
            self.flame_masks = pickle.load(f, encoding="latin1")

        if not render_full_head:
            self.final_mask = self.flame_masks["face"].tolist()
            faces = _keep_vertices_and_update_faces(faces[0], self.final_mask).unsqueeze(0)
            colors = colors[:, self.final_mask, :]

        self.register_buffer("faces", faces)
        self.register_buffer("face_colors", face_vertices(colors, faces))

    def forward(self, vertices: torch.Tensor, cam_params: torch.Tensor, **landmarks: torch.Tensor) -> dict:
        """vertices: (B, 5023, 3), cam_params: (B, 3) = [scale, tx, ty] (Sec 2.2's
        camera token, sliced at CAMERA_SCALE_SLICE + CAMERA_TRANSLATION_SLICE), and
        any number of named 3D landmark sets (e.g. landmarks_fan=..., landmarks_mp=...
        from model.flame.flame.FLAME's output) to project alongside the mesh.

        Returns: rendered_img (B,3,H,W) grayscale-shaded mesh render, transformed_vertices
        (B,5023,3, in the rasterizer's NDC-like space), and transformed_<key> (B,N,2)
        2D-projected coordinates for each landmark set passed in."""
        transformed_vertices = batch_orth_proj(vertices, cam_params)
        transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]

        transformed_landmarks = {}
        for key, points in landmarks.items():
            projected = batch_orth_proj(points, cam_params)
            projected[:, :, 1:] = -projected[:, :, 1:]
            transformed_landmarks[f"transformed_{key}"] = projected[..., :2]

        rendered_img = self._render(vertices, transformed_vertices)

        outputs = {"rendered_img": rendered_img, "transformed_vertices": transformed_vertices}
        outputs.update(transformed_landmarks)
        return outputs

    def _render(self, vertices: torch.Tensor, transformed_vertices: torch.Tensor) -> torch.Tensor:
        batch_size = vertices.shape[0]

        light_positions = (
            torch.tensor([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1], [0, 0, 1]])[None, :, :]
            .expand(batch_size, -1, -1)
            .float()
        )
        light_intensities = torch.ones_like(light_positions) * 1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)

        if not self.render_full_head:
            transformed_vertices = transformed_vertices[:, self.final_mask, :]
            vertices = vertices[:, self.final_mask, :]

        # rasterizer expects near=0, far=100 - shift the mesh so min z > 0.
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        faces = self.faces.expand(batch_size, -1, -1)
        normals = vertex_normals(vertices, faces)
        face_normals = face_vertices(normals, faces)
        colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([colors, face_normals], -1)

        rendering = self._rasterize(transformed_vertices, faces, attributes)
        albedo_images = rendering[:, :3, :, :]
        normal_images = rendering[:, 3:6, :, :]

        shading = self._add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = (
            shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3])
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return albedo_images * shading_images

    def _rasterize(self, vertices: torch.Tensor, faces: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]

        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, _, bary_coords, _ = rasterize_meshes(
            meshes_screen,
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )
        vismask = (pix_to_face > -1).float()
        dim = attributes.shape[-1]
        attributes = attributes.clone().view(attributes.shape[0] * attributes.shape[1], 3, dim)
        n, h, w, k, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(n * h * w * k, 1, 1).expand(n * h * w * k, 3, dim)
        pixel_face_vals = attributes.gather(0, idx).view(n, h, w, k, 3, dim)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        return torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)

    def _add_directionlight(self, normals: torch.Tensor, lights: torch.Tensor) -> torch.Tensor:
        """normals: (B, N, 3), lights: (B, num_lights, 6) = [direction(3), intensity(3)]
        -> (B, N, 3) shading (mean over lights of clamped Lambertian dot product)."""
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(
            light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3
        )
        normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0.0, 1.0)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)
