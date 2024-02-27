import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import transform
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import Unpack

from data.typings import BaseDatasetKwargs, BaseDataTuple, DataPathList, DatasetModes, T


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
        self.max_items = kwargs.get("max_items")
        self.seed = kwargs.get("seed")
        self.split_val_size = kwargs.get("split_val_size", 0)
        self.split_val_fold = kwargs.get("split_val_fold", 0)
        self.split_test_size = kwargs.get("split_test_size", 0)
        self.split_test_fold = kwargs.get("split_test_fold", 0)
        self.cache_data = kwargs.get("cache_data", False)
        self.dataset_name = kwargs.get("dataset_name") or self.__class__.__name__
        self.class_labels = self.set_class_labels()

        # Creating list of paths.
        self.items = self.make_items()
        if len(self.items) == 0:
            raise (RuntimeError("Get 0 items, please check"))

        self.cached_items_data: list[BaseDataTuple] = []

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
    def norm(img: NDArray) -> NDArray:
        normalized = np.zeros(img.shape)
        if len(img.shape) == 2:
            normalized = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                normalized[:, :, b] = (img[:, :, b] - img[:, :, b].mean()) / img[
                    :, :, b
                ].std()
        return normalized.astype(np.float32)

    @staticmethod
    def ensure_channels(img: NDArray) -> NDArray:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    @staticmethod
    def prepare_img_as_tensor(img: NDArray) -> Tensor:
        new_img = BaseDataset.norm(img)
        new_img = BaseDataset.ensure_channels(new_img)
        new_img = torch.from_numpy(new_img)
        return new_img

    @staticmethod
    def prepare_msk_as_tensor(msk: NDArray) -> Tensor:
        new_msk = torch.from_numpy(msk).type(torch.int64)
        return new_msk

    @staticmethod
    def resize_image(img: NDArray, resize_to: tuple[int, ...], is_mask: bool):
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
    def prepare_image_as_tensor(img: NDArray) -> Tensor:
        new_img = BaseDataset.norm(img)
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
    def extend_data(data: list[T], num_items: int, seed: int | None = None) -> list[T]:
        if len(data) >= num_items:
            return data[:num_items]
        random.seed(seed)
        extended_data = []
        new_data = data.copy()
        for i in range(num_items // len(data)):
            if i != 0:
                random.shuffle(new_data)
            extended_data.extend(new_data)
        new_data = random.sample(data, num_items - len(extended_data))
        extended_data.extend(new_data)
        random.seed(None)
        return extended_data

    @staticmethod
    def split_train_test(
        data: list[T],
        test_size: int,
        random_state: int | None = None,
        shuffle: bool = False,
        fold: int = 0,
    ) -> tuple[list[T], list[T]]:
        if shuffle:
            random.seed(random_state)
            random.shuffle(data)
            random.seed(None)
        if (fold + 1) * test_size > len(data):
            raise ValueError("Fold value is too large")
        ts = data[fold * test_size : (fold + 1) * test_size]
        tr = data[: fold * test_size] + data[(fold + 1) * test_size :]
        return tr, ts

    def make_items(self) -> DataPathList:
        all_data = self.get_all_data_path()
        test_size = round(self.split_test_size * len(all_data))
        val_size = round(self.split_val_size * len(all_data))

        tr_val, ts = self.split_train_test(
            all_data,
            test_size=test_size,
            random_state=self.seed,
            shuffle=True,
            fold=self.split_test_fold,
        )
        if self.mode == "test":
            return ts[: self.max_items]

        if self.split_test_size == 1:
            return []

        tr, val = self.split_train_test(
            tr_val,
            test_size=val_size,
            random_state=self.seed,
            shuffle=True,
            fold=self.split_val_fold,
        )
        if self.mode == "train":
            return tr[: self.max_items]
        if self.mode == "val":
            return val[: self.max_items]

        return []

    def get_data(self, index: int) -> BaseDataTuple:
        if self.cache_data and len(self.cached_items_data) > index:
            return self.cached_items_data[index]

        img_path, msk_path = self.items[index]

        img, msk = self.read_image_mask(img_path, msk_path)

        img = self.resize_image(img, self.resize_to, False)
        msk = self.resize_image(msk, self.resize_to, True)

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
