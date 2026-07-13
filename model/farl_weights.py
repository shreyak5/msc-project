"""Load FaRL-B's official pretrained ViT-B/16 weights into model.encoder.SViT
(implementation-plan.md Sec 2.1: "Initialized from FaRL-B pretrained weights").

Checkpoint: FaRL-Base-Patch16-LAIONFace20M-ep64.pth, official release from
https://github.com/FacePerceiver/FaRL (Zheng et al., CVPR 2022, MIT License,
Copyright (c) Microsoft Corporation).

FaRL's checkpoint reuses OpenAI CLIP's ViT-B/16 verbatim; its "visual.*"-prefixed
keys are (verified against the actual checkpoint):
  conv1, class_embedding, positional_embedding, ln_pre,
  transformer.resblocks.{0..11}.{ln_1,ln_2,attn,mlp}, ln_post, proj
plus an unrelated masked-image-modeling branch (lm_transformer, lm_head, ln_lm,
mask_token) that this loader ignores - it's an auxiliary training-only head, not
part of the visual encoder we initialize from.

positional_embedding has 197 rows: row 0 is the CLS token's position embedding,
rows 1-196 are the 14x14 patch grid's. SViT has no CLS token, so only row 0 is
dropped - the 196 patch rows ARE carried over and used to initialize SViT's
pos_embed (trainable from there, per Sec 2.1).

Not carried over at all: class_embedding (the CLS token itself - SViT has none)
and proj (CLIP's image-text alignment head, irrelevant here).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.encoder import SViT

_FARL_IMG_SIZE = 224
_FARL_PATCH_SIZE = 16


def _interpolate_pos_embed(
    pos_embed: torch.Tensor, old_grid: tuple[int, int], new_grid: tuple[int, int]
) -> torch.Tensor:
    """pos_embed: (old_grid[0] * old_grid[1], dim) -> (new_grid[0] * new_grid[1], dim).
    Bicubic resample of the 2D patch grid (implementation-plan.md Sec 2.1); a no-op
    when old_grid == new_grid, which is the case at FaRL's native 224x224/patch16."""
    if old_grid == new_grid:
        return pos_embed
    dim = pos_embed.shape[-1]
    grid = pos_embed.reshape(1, old_grid[0], old_grid[1], dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=new_grid, mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(new_grid[0] * new_grid[1], dim)


def load_farl_pretrained(model: SViT, checkpoint_path: str) -> torch.nn.modules.module._IncompatibleKeys:
    """Loads FaRL-B weights into `model` in place. Only touches patch_embed, pos_embed,
    ln_pre, blocks[i], and norm - component_tokens and heads are left at their own
    (from-scratch) initialization, since FaRL has no equivalent for them. Returns
    load_state_dict's IncompatibleKeys so callers/tests can verify exactly what was
    (and wasn't) loaded."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state_dict = checkpoint["state_dict"]
    visual = {k[len("visual.") :]: v for k, v in raw_state_dict.items() if k.startswith("visual.")}

    cfg = model.config
    new_grid = (cfg.img_size // cfg.patch_size, cfg.img_size // cfg.patch_size)
    farl_grid = (_FARL_IMG_SIZE // _FARL_PATCH_SIZE, _FARL_IMG_SIZE // _FARL_PATCH_SIZE)

    mapped: dict[str, torch.Tensor] = {}
    mapped["patch_embed.proj.weight"] = visual["conv1.weight"]

    patch_pos_embed = visual["positional_embedding"][1:]  # drop FaRL's CLS row (index 0) only
    mapped["pos_embed"] = _interpolate_pos_embed(patch_pos_embed, farl_grid, new_grid).unsqueeze(0)

    mapped["ln_pre.weight"] = visual["ln_pre.weight"]
    mapped["ln_pre.bias"] = visual["ln_pre.bias"]

    for i in range(cfg.depth):
        src = f"transformer.resblocks.{i}."
        dst = f"blocks.{i}."
        for suffix in (
            "ln_1.weight",
            "ln_1.bias",
            "ln_2.weight",
            "ln_2.bias",
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ):
            mapped[dst + suffix] = visual[src + suffix]

    mapped["norm.weight"] = visual["ln_post.weight"]
    mapped["norm.bias"] = visual["ln_post.bias"]

    own_state = model.state_dict()
    for key, value in mapped.items():
        if key in own_state:
            assert own_state[key].shape == value.shape, (
                f"shape mismatch for {key}: model has {tuple(own_state[key].shape)}, "
                f"checkpoint has {tuple(value.shape)}"
            )

    return model.load_state_dict(mapped, strict=False)
