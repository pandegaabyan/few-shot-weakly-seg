import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import transform
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    def __init__(self,
                 num_classes: int,
                 resize_to: tuple[int, int]
                 ):

        # Initializing variables.
        self.num_classes = num_classes
        self.resize_to = resize_to

        # Creating list of paths.
        self.items = self.make_data_list()
        if len(self.items) == 0:
            raise (RuntimeError('Found 0 items, please check the dataset'))

    # Function that create the list of pairs (img_path, mask_path)
    # Implement this function for your dataset structure
    @abstractmethod
    def get_all_data_path(self) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def make_data_list(self) -> list[tuple[str, str]]:
        pass

    @abstractmethod
    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        # image should have shape (H, W, C) and dtype uint8 while mask should have shape (H, W) and dtype int8
        pass

    @staticmethod
    def norm(img: NDArray) -> NDArray:
        normalized = np.zeros(img.shape)
        if len(img.shape) == 2:
            normalized = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                normalized[:, :, b] = (img[:, :, b] - img[:, :, b].mean()) / img[:, :, b].std()
        return normalized.astype(np.float32)

    @staticmethod
    def ensure_channels(img: NDArray) -> NDArray:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    @staticmethod
    def prepare_img_as_tensor(img: NDArray) -> torch.Tensor:
        new_img = BaseDataset.norm(img)
        new_img = BaseDataset.ensure_channels(new_img)
        new_img = torch.from_numpy(new_img)
        return new_img

    @staticmethod
    def prepare_msk_as_tensor(msk: NDArray) -> torch.Tensor:
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

        resized = transform.resize(img, resize_to, order=order,
                                   preserve_range=True, anti_aliasing=anti_aliasing)
        resized = resized.astype(img.dtype)

        return resized

    @staticmethod
    def filename_from_path(path: str) -> str:
        # Splitting path.
        filename = os.path.split(path)[-1]

        # remove extension from filename
        filename = ".".join(filename.split(".")[:-1])

        return filename

    # Function to load images and masks
    # Implement this function based on your data
    # Returns: img, mask, img_filename
    def get_data(self, index: int) -> tuple[NDArray, NDArray, str]:
        img_path, msk_path = self.items[index]

        img, msk = self.read_image_mask(img_path, msk_path)

        img = self.resize_image(img, self.resize_to, False)
        msk = self.resize_image(msk, self.resize_to, True)

        img_filename = self.filename_from_path(img_path)

        return img, msk, img_filename
