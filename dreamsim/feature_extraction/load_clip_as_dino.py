import os
from os import PathLike
from pathlib import Path

import torch

from .vision_transformer import VisionTransformer, vit_base


class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def load_clip_as_dino(patch_size: int, load_dir: PathLike = "./models", l14: bool = False):
    load_dir = Path(load_dir).resolve()
    if l14:
        sd = torch.load(load_dir / "clipl14_as_dino_vitl.pth.tar", map_location="cpu")
        dino_vit = VisionTransformer(**sd["kwargs"])
        sd = sd["state_dict"]
    else:
        sd = torch.load(load_dir / f"clip_vitb{patch_size}_pretrain.pth.tar", map_location="cpu")
        sd = sd["state_dict"]
        dino_vit = vit_base(patch_size=patch_size)

    dino_vit.pos_drop = torch.nn.LayerNorm(dino_vit.embed_dim)
    proj = sd.pop("proj")
    dino_vit.load_state_dict(sd)

    # GeLU -> QuickGeLU
    for blk in dino_vit.blocks:
        blk.mlp.act = QuickGELU()

    # LN eps 1e-6 -> 1e-5
    for m in dino_vit.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = 1e-5

    return dino_vit, proj
