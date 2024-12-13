import argparse
import logging
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


class multimodal_vit(nn.Module):
    def __init__(
        self,
        archs: Dict,
        args: argparse.Namespace,
        fuse_alg: Literal["linear", "xattn"] = "linear",
        teacher: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        archs: Dict
            The dictionary containing the model name and model architecture, e.g., {"s1": "vit_base"}. The architecture can be specified in the config file. By default, four archtectures are provided: vit_small, vit_base, vit_large and vit_giant2, see "./vision_transformer.py" for more details
        args: argparse.Namespace
            All arguments in the config file
        fuse_alg: Literal["linear", "xattn"]
            The way to fuse different modalities. By default we use linear projection
        teacher: bool
            Whether it is the teacher network, by default False
        """
        super().__init__()
        self.fuse_alg = fuse_alg
        self.args = args
        self.modal_dict = nn.ModuleDict()
        self.embed_dim = []
        for (modal_name, modal_arch), modal_args in zip(archs.items(), kwargs.values()):
            if teacher:
                model = vits.__dict__[modal_arch](**modal_args)
            else:
                model = vits.__dict__[modal_arch](
                    **modal_args,
                    drop_path_rate=args.drop_path_rate,
                    drop_path_uniform=args.drop_path_uniform,
                )
            self.modal_dict[modal_name] = model
            self.embed_dim.append(model.embed_dim)
        if fuse_alg == "linear":
            self.linear_fuse = nn.Sequential(
                nn.Linear(sum(self.embed_dim), 2 * sum(self.embed_dim)),
                nn.LayerNorm(2 * sum(self.embed_dim)),
                nn.GELU(),
                nn.Linear(2 * sum(self.embed_dim), self.embed_dim[0]),
            )
        else:
            self.nlayer = args.xattn.nlayer
            self.linear1 = nn.Linear(self.embed_dim[0], self.embed_dim[0] * 4)
            self.dropout = nn.Dropout(args.xattn.dropout)
            self.linear2 = nn.Linear(self.embed_dim[0] * 4, self.embed_dim[0])
            self.norm1 = nn.LayerNorm(self.embed_dim[0])
            self.norm2 = nn.LayerNorm(self.embed_dim[0])
            self.norm3 = nn.LayerNorm(self.embed_dim[0])
            self.dropout1 = nn.Dropout(args.xattn.dropout)
            self.dropout2 = nn.Dropout(args.xattn.dropout)
            self.dropout3 = nn.Dropout(args.xattn.dropout)
            self.activation = nn.GELU()
            self.attn = nn.MultiheadAttention(self.embed_dim[0], num_heads=args.xattn.nhead, dropout=args.xattn.dropout)

    def cross_attention(self, q, k, v):
        res1 = q
        res2 = self.attn(query=q, key=k, value=v)[0]
        res1 = res1 + self.dropout1(res2)
        res1 = self.norm1(res1)
        res2 = self.linear2(self.dropout(self.activation(self.linear1(res1))))
        res1 = res1 + self.dropout3(res2)
        res1 = self.norm3(res1)
        return res1

    def forward(self, s1, s2, is_training=True, tag="teacher", doy=None, masks=None):
        s1_out = self.modal_dict["s1"](s1, is_training=is_training, tag=tag, doy=doy, masks=masks)
        s2_out = self.modal_dict["s2"](s2, is_training=is_training, tag=tag, doy=doy, masks=masks)
        if tag == "teacher":
            out = {}
            if self.fuse_alg == "linear":
                out["x_norm"] = self.linear_fuse(torch.cat([s1_out["x_norm"], s2_out["x_norm"]], dim=-1))
                out["x_norm_clstoken"] = out["x_norm"][:, 0]
                out["x_norm_patchtokens"] = out["x_norm"][:, self.args.num_register_tokens + 1 :]
            else:
                res = s1_out["x_norm"]
                for _ in range(self.nlayer):
                    q = res
                    res = self.cross_attention(q, s2_out["x_norm"], s2_out["x_norm"])
                out["x_norm_clstoken"] = res[:, 0]
                out["x_norm_patchtokens"] = res[:, self.args.num_register_tokens + 1 :]
            return out
        else:
            out_global, out_local = {}, {}
            if self.fuse_alg == "linear":
                out_global["x_norm"] = self.linear_fuse(torch.cat([s1_out[0]["x_norm"], s2_out[0]["x_norm"]], dim=-1))
                out_global["x_norm_clstoken"] = out_global["x_norm"][:, 0]
                out_global["x_norm_patchtokens"] = out_global["x_norm"][:, self.args.num_register_tokens + 1 :]
                out_local["x_norm"] = self.linear_fuse(torch.cat([s1_out[1]["x_norm"], s2_out[1]["x_norm"]], dim=-1))
                out_local["x_norm_clstoken"] = out_local["x_norm"][:, 0]
                out_local["x_norm_patchtokens"] = out_local["x_norm"][:, self.args.num_register_tokens + 1 :]
            else:
                res_global = s1_out[0]["x_norm"]
                res_local = s1_out[1]["x_norm"]
                for _ in range(self.nlayer):
                    q_global = res_global
                    res_global = self.cross_attention(q_global, s2_out[0]["x_norm"], s2_out[0]["x_norm"])
                    q_local = res_local
                    res_cls_local = self.cross_attention(q_local, s2_out[1]["x_norm"], s2_out[1]["x_norm"])
                    q_local = res_local
                    res_local = self.cross_attention(q_local, s2_out[1]["x_norm"], s2_out[1]["x_norm"])
                out_global["x_norm_clstoken"] = res_global[:, 0]
                out_global["x_norm_patchtokens"] = res_global[:, self.args.num_register_tokens + 1 :]
                out_local["x_norm_clstoken"] = res_local[:, 0]
                out_local["x_norm_patchtokens"] = res_local[:, self.args.num_register_tokens + 1 :]
            return out_global, out_local
