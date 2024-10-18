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


class ClusterSentinel2Dataset(PretrainSentinel2Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        split: "ClusterSentinel2Dataset.Split",  # this is called Forward reference, when we initialize, Split does not exist
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)

    def __getitem__(self, index) -> None:
        tiles, target = self.file_lst[index]
        image = self._composite(tiles)
        if isinstance(image, np.ndarray):
            transform_2_tensor = transforms.Compose([
                transforms.v2.ToImageTensor(), 
                transforms.ConvertImageDtype(torch.float)])
            image = transform_2_tensor(image)
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
            if not files or (files and dirs):
                continue
            img_needed = files if len(files) == len(band) else None
            if img_needed:
                label = CATEGORY_LABEL_DICT[rts.split("/")[-3]]
                img_needed = sorted(img_needed, key=lambda x: list(BAND_ORDER.keys()).index(x.split("_")[0]))
                img_needed = [os.path.join(rts, img) for img in img_needed]
                file_lst.append((img_needed, label))
        # Export file_lst to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(file_lst, json_file, indent=4)
            
    @staticmethod
    def cluster_by_label(root, save_dir):
        import collections
        from tqdm import tqdm

        pbar = tqdm(total=len(os.listdir(root)))

        def update_pbar(arg=None):
            pbar.update(1)

        def execute(rts, save_dirs):
            target = os.path.join(rts, "land_cover.png")
            lc = imread(target)
            frequency = collections.Counter(lc.flatten())
            if frequency[list(frequency.keys())[0]] > 512 * 256:
                max_lc = LANDCOVER[list(frequency.keys())[0]]
                target_dir = os.path.join(save_dir, max_lc)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copytree(rts, os.path.join(target_dir, rts.split("/")[-1]))

        pool = ThreadPool(60)
        for rts, dirs, files in os.walk(root):
            if not files:
                continue
            if "land_cover.png" in files:
                pool.apply_async(execute, args=[rts, save_dir], error_callback=print, callback=update_pbar)
                continue
        pool.close()
        pool.join()

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
    # split_dataset(
    #     "/NAS6/Members/yufengyang/Satlas/label_landcover", "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas"
    # )
    # remove_empty("/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train")
    do_statistic("/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters/test_all.json")
    # dump_entries(
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters/train",
    #     ["tci", "b05", "b06", "b07", "b08", "b11", "b12"],
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters/train_all.json",
    # )
    # cluster_by_label(
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train",
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters_train",
    # )
