from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


class multimodal_vit(nn.Module):
    def __init__(self, archs, **kwargs):
        super().__init__()
        self.modal_dict = nn.ModuleDict()
        self.embed_dim = []
        for (modal_name, modal_arch), modal_args in zip(archs.items(), kwargs.values()):
            model = vits.__dict__[modal_arch](**modal_args)
            self.modal_dict[modal_name] = model
            self.embed_dim.append(model.embed_dim)
        self.cls_fuse = nn.Linear(sum(self.embed_dim), self.embed_dim[0])
        self.patch_fuse = nn.Linear(sum(self.embed_dim), self.embed_dim[0])

    def forward(self, s1, s2, is_training=True, tag="teacher", doy=None, masks=None):
        s1_out = self.modal_dict["s1"](s1, is_training=is_training, tag=tag, doy=doy, masks=masks)
        s2_out = self.modal_dict["s2"](s2, is_training=is_training, tag=tag, doy=doy, masks=masks)
        if tag == "teacher":
            out = {}
            out["x_norm_clstoken"] = self.cls_fuse(
                torch.cat([s1_out["x_norm_clstoken"], s2_out["x_norm_clstoken"]], dim=-1)
            )
            out["x_norm_patchtokens"] = self.patch_fuse(
                torch.cat([s1_out["x_norm_patchtokens"], s2_out["x_norm_patchtokens"]], dim=-1)
            )
            return out
        else:
            out_global, out_local = {}, {}
            out_global["x_norm_clstoken"] = self.cls_fuse(
                torch.cat([s1_out[0]["x_norm_clstoken"], s2_out[0]["x_norm_clstoken"]], dim=-1)
            )
            out_global["x_norm_patchtokens"] = self.patch_fuse(
                torch.cat([s1_out[0]["x_norm_patchtokens"], s2_out[0]["x_norm_patchtokens"]], dim=-1)
            )
            out_local["x_norm_clstoken"] = self.cls_fuse(
                torch.cat([s1_out[1]["x_norm_clstoken"], s2_out[1]["x_norm_clstoken"]], dim=-1)
            )
            out_local["x_norm_patchtokens"] = self.patch_fuse(
                torch.cat([s1_out[1]["x_norm_patchtokens"], s2_out[1]["x_norm_patchtokens"]], dim=-1)
            )
            return out_global, out_local
