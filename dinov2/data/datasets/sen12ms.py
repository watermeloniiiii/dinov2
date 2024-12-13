# @author: Chenxi Lin

from enum import Enum
import numpy as np
import json
import os
import rasterio
from shutil import copyfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, List, Optional, Tuple, Union, Any

NUM_DAY_PER_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 258045,
            _Split.VAL: 110757,
            _Split.TEST: 0,
        }
        return split_lengths[self]


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
        doy = 0
        for m in range(1, month):
            doy += NUM_DAY_PER_MONTH[m]
        doy += day
        return doy

    def __getitem__(self, index) -> None:
        s1_tile, s2_tile = self.file_lst[index]
        date = self._str_2_doy(s1_tile.split("/")[-1].split("_")[5])
        s1 = rasterio.open(s1_tile).read().transpose(1, 2, 0)
        s2 = rasterio.open(s2_tile).read().astype(np.float32).transpose(1, 2, 0)
        if isinstance(s1, np.ndarray):
            transform_2_tensor = transforms.Compose(
                [transforms.v2.ToImageTensor(), transforms.ConvertImageDtype(torch.float)]
            )
            s1 = transform_2_tensor(s1)
            s2 = transform_2_tensor(s2)
        if self.transforms is not None:
            s1, s2, target = self.transforms(s1, s2, 1)
        return s1, s2, target, date


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
    """
    Make training/validation dataset from original data
    """

    for type in ["train", "val"]:
        for modal in ["s1", "s2"]:
            target_folder = os.path.join(save_dir, type, modal)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)
    for root, _, files in os.walk(root):
        if files and files[0].endswith(".tif"):
            modality = files[0][:2]
            if modality == "s2":
                continue
            counterpart_root = root.replace("S1", "S2")
            # NOTE os.listdir() returns unordered list so have to do sorting
            ordered_s2_lst = sorted(os.listdir(counterpart_root), key=lambda x: int(x.split(".")[0].split("_")[-1]))
            for tile in files:
                index = tile.split(".")[0].split("_")[-1]

                # ----- find S2 counterpart -----#
                """The reason for this searching is that S1 and S2 have different acquisition date and so we cannot simply replace S1 with S2 but have to find corresonding index"""
                counterpart = ordered_s2_lst[int(index)]
                # --------------------------------#

                # 70% training and 30% validation
                random_flag = np.random.choice(2, 1, p=[0.7, 0.3])
                SPLIT_INDEX = {0: "train", 1: "val"}
                copyfile(os.path.join(root, tile), os.path.join(save_dir, SPLIT_INDEX[random_flag], modality, tile))
                if counterpart.split(".")[0].split("_")[-1] != index:
                    raise ValueError("S1 and S2 data do not match!")
                copyfile(
                    os.path.join(counterpart_root, counterpart),
                    os.path.join(save_dir, SPLIT_INDEX[random_flag], "s2", counterpart),
                )


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
    # Step 1. transfer the data from NAS to the target folder and split into training and validation sets
    split_dataset(
        "/NAS/datasets/PUBLIC_DATASETS/SEN12MS-CR-TS/",
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms",
    )

    # Step 2. Store all samples into json files
    for dataset in ["train", "val"]:
        dump_entries(
            f"/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/{dataset}/s1",
            f"/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/{dataset}_all.json",
        )
