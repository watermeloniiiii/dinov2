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
        self.modal_fuse = nn.Linear(sum(self.embed_dim), self.embed_dim[0])

    def forward(self, s1, s2):
        s1_out = self.modal_dict["s1"](s1)
        s2_out = self.modal_dict["s2"](s2)
        out = self.modal_fuse(torch.cat([s1_out, s2_out], dim=-1))
        return out
