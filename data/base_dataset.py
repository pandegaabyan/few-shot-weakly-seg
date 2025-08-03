import os
import random
from abc import ABC, abstractmethod
from math import floor

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray
from skimage import transform
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import Unpack

from data.typings import (
    BaseDatasetKwargs,
    BaseDataTuple,
    DataPathList,
    DatasetModes,
    ScalingType,
    T,
)


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        mode: DatasetModes,
        num_classes: int,
        resize_to: tuple[int, int],
        **kwargs: Unpack[BaseDatasetKwargs],
    ):
        # Initializing variables.
        self.mode = mode
        self.num_classes = num_classes
        self.resize_to = resize_to
        self.dataset_name = kwargs.get("dataset_name") or self.__class__.__name__
        self.size = kwargs.get("size", 1.0)
        self.split_val_size = kwargs.get("split_val_size", 0)
        self.split_val_fold = kwargs.get("split_val_fold", 0)
        self.split_test_size = kwargs.get("split_test_size", 0)
        self.split_test_fold = kwargs.get("split_test_fold", 0)
        self.scaling: ScalingType = kwargs.get("scaling", None)
        self.cache_data = kwargs.get("cache_data", False)
        self.class_labels = self.set_class_labels()
        self.seed = kwargs.get("seed", 0) * int(1e4) + self.str_to_num(
            self.dataset_name
        )

        transforms = kwargs.get("transforms", None)
        if transforms == "basic":
            self.transforms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
                    A.GaussianBlur(blur_limit=4, p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        ensure_safe_range=True,
                        p=0.5,
                    ),
                ]
            )
        else:
            self.transforms = transforms

        # Creating list of paths.
        self.items = self.make_items()
        if len(self.items) == 0:
            raise (RuntimeError("Get 0 items, please check"))

        self.cached_items_data: list[BaseDataTuple] = []

        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    # Function that create the list of pairs (img_path, mask_path)
    # Implement this function for your dataset structure
    @abstractmethod
    def get_all_data_path(self) -> DataPathList:
        pass

    @abstractmethod
    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        # image should have shape (H, W, C) and dtype uint8 while mask should have shape (H, W) and dtype int8
        pass

    @abstractmethod
    def set_class_labels(self) -> dict[int, str]:
        pass

    @staticmethod
    def str_to_num(s: str) -> int:
        return sum(i * ord(c) for i, c in enumerate(s, start=1))

    @staticmethod
    def scale_min_max(img: NDArray) -> NDArray:
        scaled = np.zeros(img.shape)
        if len(img.shape) == 2:
            scaled = (img - img.min()) / (img.max() - img.min())
        else:
            for c in range(img.shape[2]):
                chan = img[:, :, c]
                scaled[:, :, c] = (chan - chan.min()) / (chan.max() - chan.min())
        return scaled.astype(np.float32)

    @staticmethod
    def scale_mean_std(img: NDArray) -> NDArray:
        scaled = np.zeros(img.shape)
        if len(img.shape) == 2:
            scaled = (img - img.mean()) / img.std()
        else:
            for c in range(img.shape[2]):
                chan = img[:, :, c]
                scaled[:, :, c] = (chan - chan.mean()) / chan.std()
        return scaled.astype(np.float32)

    @staticmethod
    def ensure_channels(img: NDArray) -> NDArray:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    @staticmethod
    def resize_image(
        img: NDArray, resize_to: tuple[int, ...], is_mask: bool
    ) -> NDArray:
        if is_mask:
            order = 0
            anti_aliasing = False
        else:
            order = 1
            anti_aliasing = True

        resized = transform.resize(
            img,
            resize_to,
            order=order,
            preserve_range=True,
            anti_aliasing=anti_aliasing,
        )
        resized = resized.astype(img.dtype)

        return resized

    @staticmethod
    def prepare_image_as_tensor(img: NDArray, scaling: ScalingType = None) -> Tensor:
        if scaling == "simple":
            new_img = (img / 255.0).astype(np.float32)
        elif scaling == "min-max":
            new_img = BaseDataset.scale_min_max(img)
        elif scaling == "mean-std":
            new_img = BaseDataset.scale_mean_std(img)
        else:
            new_img = img
        new_img = BaseDataset.ensure_channels(new_img)
        new_img = torch.from_numpy(new_img)
        return new_img

    @staticmethod
    def prepare_mask_as_tensor(msk: NDArray) -> Tensor:
        new_msk = torch.from_numpy(msk).type(torch.int64)
        return new_msk

    @staticmethod
    def filename_from_path(path: str) -> str:
        # Splitting path.
        filename = os.path.split(path)[-1]

        # remove extension from filename
        filename = ".".join(filename.split(".")[:-1])

        return filename

    @staticmethod
    def extend_data(
        data: list[T],
        num_items: int,
        random_state: int | None = None,
    ) -> list[T]:
        if len(data) >= num_items:
            return data[:num_items]

        extended_data = []
        new_data = data.copy()
        if random_state is not None:
            rng = random.Random(random_state)

        for i in range(num_items // len(data)):
            if i != 0 and random_state is not None:
                rng.shuffle(new_data)
            extended_data.extend(new_data)
            if random_state is not None:
                new_data = rng.sample(data, num_items - len(extended_data))
            else:
                new_data = data[: num_items - len(extended_data)]
            extended_data.extend(new_data)

        return extended_data

    @staticmethod
    def split_train_test(
        data: list[T],
        test_size: int,
        random_state: int | None = None,
        fold: int = 0,
    ) -> tuple[list[T], list[T]]:
        if test_size == 0:
            return data, []
        if random_state is not None:
            rng = random.Random(random_state)
            rng.shuffle(data)
        if (fold + 1) * test_size > len(data):
            raise ValueError("Fold value is too large")
        ts = data[fold * test_size : (fold + 1) * test_size]
        tr = data[: fold * test_size] + data[(fold + 1) * test_size :]
        return tr, ts

    def make_items(self) -> DataPathList:
        def finalize(data: list) -> DataPathList:
            if isinstance(self.size, int):
                num_items = self.size
            else:
                num_items = floor(self.size * len(data))
            return self.extend_data(data, num_items)

        all_data = self.get_all_data_path()
        test_size = floor(self.split_test_size * len(all_data))
        val_size = floor(self.split_val_size * len(all_data))

        tr_val, ts = self.split_train_test(
            all_data,
            test_size,
            random_state=self.seed + 4819,
            fold=self.split_test_fold,
        )
        if self.mode == "test":
            return finalize(ts)

        if self.split_test_size == 1:
            return []

        tr, val = self.split_train_test(
            tr_val,
            val_size,
            random_state=self.seed + 8732,
            fold=self.split_val_fold,
        )
        if self.mode == "train":
            return finalize(tr)
        if self.mode == "val":
            return finalize(val)

        return []

    def get_data(self, index: int) -> BaseDataTuple:
        if self.cache_data and len(self.cached_items_data) > index:
            return self.cached_items_data[index]

        img_path, msk_path = self.items[index]

        img, msk = self.read_image_mask(img_path, msk_path)

        img = self.resize_image(img, self.resize_to, False)
        msk = self.resize_image(msk, self.resize_to, True)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]

        img_filename = self.filename_from_path(img_path)

        data = BaseDataTuple(
            image=img,
            mask=msk,
            file_name=img_filename,
        )

        return data

    def fill_cached_items_data(self):
        if not self.cache_data:
            return

        for i in range(len(self.items)):
            self.cached_items_data.append(self.get_data(i))
