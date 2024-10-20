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
from io import BytesIO
from typing import Any

from PIL import Image
import numpy as np
import shutil
from multiprocessing.pool import ThreadPool

BAND_ORDER = {"tci": 0, "b05": 0, "b06": 0, "b07": 0, "b08": 0, "b11": 0, "b12": 0}
BANDS = {"tci": (480427, 0, 52255), "all": (480329, 0, 52247)}

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


class PretrainSentinel2Dataset(Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        split: "PretrainSentinel2Dataset.Split",  # this is called Forward reference, when we initialize, Split does not exist
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(self.root, split.value + "_all.json"), "r") as file:
            self.file_lst = json.load(file)

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        self._split = split

    def __len__(self):
        return len(self.file_lst)

    def _str_2_doy(self, date):
        month = int(date[:2])
        day = int(date[2:])
        num_day_per_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        doy = 0
        for m in range(1, month):
            doy += num_day_per_month[m]
        doy += day
        return doy

    def __getitem__(self, index) -> None:
        tiles = self.file_lst[index]
        date = self._str_2_doy(tiles[0].split("/")[-2][-4:])
        image = self._composite(tiles)
        if isinstance(image, np.ndarray):
            transform_2_tensor = transforms.Compose(
                [transforms.v2.ToImageTensor(), transforms.ConvertImageDtype(torch.float)]
            )
            image = transform_2_tensor(image)
        if self.transforms is not None:
            image, target = self.transforms(image, 1)
        return image, target, date

    def _composite(self, dirs):
        res = []
        for dir in dirs:
            img = np.array(Image.open(dir))
            if len(img.shape) == 2:
                img = img[..., None]
            res.append(img)
        return np.concatenate([r for r in res], axis=2)


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


def split_dataset(root, save_dir):
    from shutil import copytree

    for type in ["train", "val", "test"]:
        target_folder = os.path.join(save_dir, type)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
    for tile in os.listdir(root):
        random_flag = np.random.choice(3, 1, p=[0.9, 0, 0.1])
        if random_flag == 0:
            copytree(os.path.join(root, tile), os.path.join(save_dir, "train", tile), dirs_exist_ok=True)
        elif random_flag == 1:
            copytree(os.path.join(root, tile), os.path.join(save_dir, "val", tile), dirs_exist_ok=True)
        else:
            copytree(os.path.join(root, tile), os.path.join(save_dir, "test", tile), dirs_exist_ok=True)


def remove_empty(root):
    from shutil import rmtree

    for tile in os.listdir(root):
        if len(os.listdir(os.path.join(root, tile))) < 6:
            rmtree(os.path.join(root, tile))


def do_statistic(root, band):
    count = 0
    for _, dirs, files in os.walk(root):
        if not files or (files and dirs):
            continue
        count += 1 if len(files) == len(band) else 0
    print(count)


def dump_entries(root, band, output_file):
    file_lst = []
    for rts, dirs, files in os.walk(root):
        if not files or (files and dirs):
            continue
        img_needed = files if len(files) == len(band) else None
        if img_needed:
            img_needed = sorted(img_needed, key=lambda x: list(BAND_ORDER.keys()).index(x.split("_")[0]))
            img_needed = [os.path.join(rts, img) for img in img_needed]
            file_lst.append(img_needed)
    # Export file_lst to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(file_lst, json_file, indent=4)


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


if __name__ == "__main__":
    # split_dataset(
    #     "/NAS6/Members/yufengyang/Satlas/label_landcover", "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas"
    # )
    # remove_empty("/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train")
    # do_statistic(
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train",
    #     ["tci", "b05", "b06", "b07", "b08", "b11", "b12"],
    # )
    dump_entries(
        "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train",
        ["tci", "b05", "b06", "b07", "b08", "b11", "b12"],
        "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train_all.json",
    )
    # cluster_by_label(
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train",
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters_train",
    # )
