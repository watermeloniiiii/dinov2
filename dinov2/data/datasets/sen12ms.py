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
import rasterio


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 258045,
            _Split.VAL: 110757,
            _Split.TEST: 0,
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


class SEN12MSDataset(Dataset):
    Split = Union[_Split]

    def __init__(
        self,
        split: "SEN12MSDataset.Split",  # this is called Forward reference, when we initialize, Split does not exist
        root: str,
        s1_transform: Optional[Callable] = None,
        s2_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        # for backwards-compatibility
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.target_transform = target_transform
        with open(os.path.join(self.root, split.value + "_all.json"), "r") as file:
            self.file_lst = json.load(file)

        transforms = StandardTransform(s1_transform, s2_transform, target_transform)
        self.transforms = transforms
        self._split = split

    def __len__(self):
        return len(self.file_lst)

    def _str_2_doy(self, date):
        month = int(date[5:7])
        day = int(date[8:])
        num_day_per_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        doy = 0
        for m in range(1, month):
            doy += num_day_per_month[m]
        doy += day
        return doy

    def __getitem__(self, index) -> None:
        s1_tile, s2_tile = self.file_lst[index]
        date = self._str_2_doy(s1_tile.split("/")[-1].split("_")[5])
        s1 = rasterio.open(s1_tile).read()
        s2 = rasterio.open(s2_tile).read().astype(np.float32)
        if isinstance(s1, np.ndarray):
            transform_2_tensor = transforms.Compose(
                [transforms.v2.ToImageTensor(), transforms.ConvertImageDtype(torch.float)]
            )
            s1 = transform_2_tensor(s1)
            s2 = transform_2_tensor(s2)
        if self.transforms is not None:
            s1, s2, target = self.transforms(s1, s2, 1)
        return s1, s2, target, date

    def _composite(self, dirs):
        res = []
        for dir in dirs:
            img = np.array(Image.open(dir))
            if len(img.shape) == 2:
                img = img[..., None]
            res.append(img)
        return np.concatenate([r for r in res], axis=2)


class StandardTransform:
    def __init__(
        self,
        s1_transform: Optional[Callable] = None,
        s2_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.target_transform = target_transform

    def __call__(self, s1: Any, s2: Any, target: Any) -> Tuple[Any, Any]:
        if self.s1_transform is not None:
            s1 = self.s1_transform(s1)
        if self.s2_transform is not None:
            s2 = self.s2_transform(s2)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return s1, s2, target

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
    from shutil import copyfile

    for type in ["train", "val"]:
        for modal in ["s1", "s2"]:
            target_folder = os.path.join(save_dir, type, modal)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
    for root, _, files in os.walk(root):
        if files and files[0].endswith(".tif"):
            modality = files[0][:2]
            if modality == "s2":
                continue
            for tile in files:
                random_flag = np.random.choice(2, 1, p=[0.7, 0.3])
                if random_flag == 0:
                    copyfile(os.path.join(root, tile), os.path.join(save_dir, "train", modality, tile))
                    counterpart_root = root.replace("S1", "S2")
                    index = tile.split(".")[0].split("_")[-1]
                    counterpart = sorted(
                        os.listdir(counterpart_root), key=lambda x: int(x.split(".")[0].split("_")[-1])
                    )[int(index)]
                    if counterpart.split(".")[0].split("_")[-1] != index:
                        raise ValueError("S1 and S2 data do not match!")
                    copyfile(
                        os.path.join(counterpart_root, counterpart),
                        os.path.join(save_dir, "train", "s2", counterpart),
                    )
                else:
                    copyfile(os.path.join(root, tile), os.path.join(save_dir, "val", modality, tile))
                    counterpart_root = root.replace("S1", "S2")
                    index = tile.split(".")[0].split("_")[-1]
                    counterpart = sorted(
                        os.listdir(counterpart_root), key=lambda x: int(x.split(".")[0].split("_")[-1])
                    )[int(index)]
                    if counterpart.split(".")[0].split("_")[-1] != index:
                        raise ValueError("S1 and S2 data do not match!")
                    copyfile(
                        os.path.join(counterpart_root, counterpart),
                        os.path.join(save_dir, "val", "s2", counterpart),
                    )


def remove_empty(root):
    from shutil import rmtree

    for tile in os.listdir(root):
        if len(os.listdir(os.path.join(root, tile))) < 6:
            rmtree(os.path.join(root, tile))


def calculate_mean_variance(image_folder, channels):
    """
    Calculate the mean and variance for a dataset of multi-channel images.

    Args:
        image_folder (str): Path to the folder containing satellite images.
        channels (int): Number of channels per image.

    Returns:
        tuple: Channel-wise mean and variance arrays.
    """
    # Initialize accumulators for sum and squared sum
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_squared_sum = np.zeros(channels, dtype=np.float64)
    pixel_count = 0  # Total number of pixels across all images

    # Iterate through all images in the folder
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)

        with rasterio.open(file_path) as src:
            image = src.read()

        # Reshape to (channels, -1) for efficient computation
        image = image.reshape(image.shape[0], -1) / 10

        # Update statistics for each channel
        channel_sum += np.sum(image, axis=1)
        channel_squared_sum += np.sum(image**2, axis=1)
        pixel_count += image.shape[1]  # Total pixels in the current image

    # Compute mean and variance
    mean = channel_sum / pixel_count
    variance = (channel_squared_sum / pixel_count) - (mean**2)

    return mean, variance


def dump_entries(root, output_file):
    file_lst = []
    for file in zip(os.listdir(root), os.listdir(root.replace("s1", "s2"))):
        s1 = os.path.join(root, file[0])
        s2 = os.path.join(root.replace("s1", "s2"), file[1])
        file_lst.append([s1, s2])
    # Export file_lst to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(file_lst, json_file, indent=4)


if __name__ == "__main__":
    # split_dataset(
    #     "/NAS/datasets/PUBLIC_DATASETS/SEN12MS-CR-TS/",
    #     "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms",
    # )
    # remove_empty("/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train")
    # do_statistic(
    #     "/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/train",
    #     ["tci", "b05", "b06", "b07", "b08", "b11", "b12"],
    # )
    dump_entries(
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/val/s1",
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/val_all.json",
    )
    # dataset = SEN12MSDataset(
    #     split=SEN12MSDataset.Split["TRAIN"], root="/NAS3/Members/linchenxi/projects/foundation_model/sen12ms"
    # )
