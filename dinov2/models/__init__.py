# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from .multimodal import multimodal_vit


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, multimodal=False, img_size=224):
    if not multimodal:
        args.arch = args.arch.removesuffix("_memeff")
        if "vit" in args.arch:
            vit_kwargs = dict(
                img_size=img_size,
                in_chans=args.in_chans,
                patch_size=args.patch_size,
                init_values=args.layerscale,
                ffn_layer=args.ffn_layer,
                block_chunks=args.block_chunks,
                qkv_bias=args.qkv_bias,
                proj_bias=args.proj_bias,
                ffn_bias=args.ffn_bias,
                num_register_tokens=args.num_register_tokens,
                interpolate_offset=args.interpolate_offset,
                interpolate_antialias=args.interpolate_antialias,
            )
            teacher = vits.__dict__[args.arch](**vit_kwargs)
            if only_teacher:
                return teacher, teacher.embed_dim
            student = vits.__dict__[args.arch](
                **vit_kwargs,
                drop_path_rate=args.drop_path_rate,
                drop_path_uniform=args.drop_path_uniform,
            )
            embed_dim = student.embed_dim
    else:
        arch_s1 = args.arch_s1
        arch_s2 = args.arch_s2
        vit_kwargs_s1 = dict(
            img_size=img_size,
            in_chans=args.in_chans_s1,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        vit_kwargs_s2 = vit_kwargs_s1.copy()
        vit_kwargs_s2["in_chans"] = args.in_chans_s2
        archs = {"s1": arch_s1, "s2": arch_s2}
        teacher = multimodal_vit(archs, args, teacher=True, s1=vit_kwargs_s1, s2=vit_kwargs_s2)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = multimodal_vit(archs, args, teacher=False, s1=vit_kwargs_s1, s2=vit_kwargs_s2)
        embed_dim = student.embed_dim[0]
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student, only_teacher=only_teacher, multimodal=cfg.train.multimodal, img_size=cfg.crops.global_crops_size
    )
