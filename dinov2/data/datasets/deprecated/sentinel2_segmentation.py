# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import transforms
from imageio import imread
import json
import logging

import numpy as np
import shutil
from multiprocessing.pool import ThreadPool
from PIL import Image

from .sentinel2 import PretrainSentinel2Dataset

BAND_ORDER = {"tci": 0, "b05": 0, "b06": 0, "b07": 0, "b08": 0, "b11": 0, "b12": 0}
BANDS = {"tci": (164622, 0, 17570), "all": (164544, 0, 17563)}

LANDCOVER = {
    1: "water",
    2: "developed",
    3: "tree",
    4: "shrub",
    5: "grass",
    6: "crop",
    7: "bare",
    8: "snow",
    9: "wetland",
    10: "mangroves",
    11: "moss",
}

CATEGORY_LABEL_DICT = {
    "wetland": 0,
    "water": 1,
    "tree": 2,
    "shrub": 3,
    "grass": 4,
    "developed": 5,
    "crop": 6,
    "bare": 7,
}


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: BANDS["all"][0],
            _Split.VAL: BANDS["all"][1],
            _Split.TEST: BANDS["all"][2],
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class SegmentationSentinel2Dataset(PretrainSentinel2Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        split: "SegmentationSentinel2Dataset.Split",  # this is called Forward reference, when we initialize, Split does not exist
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)

    def __getitem__(self, index) -> None:
        tiles, target = self.file_lst[index]
        image = self._composite(tiles)
        tgt = np.array(Image.open(target))
        if isinstance(image, np.ndarray):
            transform_2_tensor_img = transforms.Compose(
                [transforms.v2.ToImageTensor(), transforms.ConvertImageDtype(torch.float)]
            )
            transform_2_tensor_tgt = transforms.Compose(
                [transforms.v2.ToImageTensor(), transforms.ConvertImageDtype(torch.LongTensor)]
            )
            image = transform_2_tensor_img(image)
            target = transform_2_tensor_tgt(target)
        if self.transforms is not None:
            image, _ = self.transforms(image, 1)
        return image, target

    @staticmethod
    def do_statistic(root):
        with open(root, "r") as file:
            file_lst = json.load(file)
        print(len(file_lst))

    @staticmethod
    def dump_entries(root, band, output_file):
        file_lst = []
        for rts, dirs, files in os.walk(root):
            # check if target exists
            if (files and dirs) and "land_cover.png" in files:
                tgt = os.path.join(rts, "land_cover.png")
            if not files or (files and dirs):
                continue
            img_needed = files if len(files) == len(band) else None
            if img_needed and tgt:
                img_needed = sorted(img_needed, key=lambda x: list(BAND_ORDER.keys()).index(x.split("_")[0]))
                img_needed = [os.path.join(rts, img) for img in img_needed]
                file_lst.append((img_needed, tgt))
        # Export file_lst to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(file_lst, json_file, indent=4)


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


if __name__ == "__main__":
    # SegmentationSentinel2Dataset.do_statistic("/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters/test_all.json")
    SegmentationSentinel2Dataset.dump_entries(
        "/NAS3/Members/linchenxi/projects/foundation_model/satlas/train",
        ["tci", "b05", "b06", "b07", "b08", "b11", "b12"],
        "/NAS3/Members/linchenxi/projects/foundation_model/satlas/train_segmentation.json",
    )
