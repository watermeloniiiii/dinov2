# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence
import math

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
SENTINEL2_DEFAULT_MEAN = (0.3584, 0.3111, 0.2654, 0.2578, 0.3299, 0.3653, 0.3547, 0.2965, 0.2266)
SENTINEL2_DEFAULT_STD = (0.064, 0.072, 0.0095, 0.048, 0.067, 0.080, 0.085, 0.074, 0.060)
SEN12MS_S2_MEAN = [
    a * 1000
    for a in list(
        [
            3.03641137,
            2.8422263,
            2.68316342,
            2.86619114,
            3.01340663,
            3.42934514,
            3.68354561,
            3.55569409,
            3.84027188,
            1.68906212,
            0.31154035,
            2.73252887,
            2.09823427,
        ]
    )
]
SEN12MS_S2_STD = [
    math.sqrt(a * 10**2)
    for a in list(
        [
            5.39760919,
            5.52136469,
            4.86646701,
            5.65607056,
            5.42180909,
            4.86437048,
            4.91524867,
            4.59927598,
            4.83957122,
            2.40673805,
            0.48222911,
            1.86288455,
            1.34343445,
        ]
    )
]

SEN12MS_S1_MEAN = [a * 10 for a in list([-1.14150634, -1.91918201])]
SEN12MS_S1_STD = [math.sqrt(a * 10**2) for a in list([0.21155498, 0.39639459])]


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def make_classification_train_transform_Sentinel2(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = SENTINEL2_DEFAULT_MEAN,
    std: Sequence[float] = SENTINEL2_DEFAULT_STD,
):
    # transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    transforms_list = []
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform_Sentinel2(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = SENTINEL2_DEFAULT_MEAN,
    std: Sequence[float] = SENTINEL2_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        # transforms.Resize(resize_size, interpolation=interpolation),
        # transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)
