import os
from abc import ABC

import numpy as np
from numpy.typing import NDArray
from skimage import io

from data.base_dataset import BaseDataset
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.types import SparsityModes, SparsityValue


def get_rim_one_data_path() -> list[tuple[str, str]]:
    data_path = "../Data/RIM-ONE"
    img_dir = "/images/"
    msk_dir = "/masks/"
    img_files = os.listdir(data_path + img_dir)
    img_files_no_ext = [
        BaseDataset.filename_from_path(img_file) for img_file in img_files
    ]
    msk_files = os.listdir(data_path + msk_dir)

    all_data_path = []
    for msk_file in msk_files:
        msk_file_no_ext = BaseDataset.filename_from_path(msk_file)
        try:
            img_index = img_files_no_ext.index(msk_file_no_ext)
            all_data_path.append(
                (
                    data_path + img_dir + img_files[img_index],
                    data_path + msk_dir + msk_file,
                )
            )
        except ValueError:
            continue

    return all_data_path


def get_drishti_data_path() -> list[tuple[str, str]]:
    data_path = "../Data/DRISHTI-GS"
    img_dir = "/images/"
    msk_dir = "/masks/"
    img_files = os.listdir(data_path + img_dir)
    img_files_no_ext = [
        BaseDataset.filename_from_path(img_file) for img_file in img_files
    ]
    msk_files = os.listdir(data_path + msk_dir)

    all_data_path = []
    for msk_file in msk_files:
        msk_file_no_ext = BaseDataset.filename_from_path(msk_file)
        try:
            img_index = img_files_no_ext.index(msk_file_no_ext)
            all_data_path.append(
                (
                    data_path + img_dir + img_files[img_index],
                    data_path + msk_dir + msk_file,
                )
            )
        except ValueError:
            continue

    return all_data_path


def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
    img = io.imread(img_path, as_gray=False)
    msk = io.imread(msk_path, as_gray=True)
    msk = (msk * 255).astype(np.int8)

    return img, msk


class OpticDiscCupDataset(FewSparseDataset, ABC):
    def set_additional_sparse_mode(self) -> list[SparsityModes]:
        return []

    def get_additional_sparse_mask(
        self,
        sparsity_mode: SparsityModes,
        msk: NDArray,
        img: NDArray = None,
        sparsity_value: SparsityValue = "random",
        seed=0,
    ) -> NDArray:
        if sparsity_mode == "point_old":
            return self.sparse_point_old(msk, self.num_classes, sparsity_value, seed)
        elif sparsity_mode == "grid_old":
            return self.sparse_grid_old(msk, sparsity_value, seed)
        else:
            return msk

    @staticmethod
    def sparse_point_old(
        msk: NDArray, num_classes: int, sparsity: SparsityValue = "random", seed=0
    ) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        # Linearizing mask.
        msk_ravel = msk.ravel()

        # Copying raveled mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros(msk_ravel.shape[0], dtype=msk.dtype)
        new_msk[:] = -1

        for c in range(num_classes):
            # Slicing array for only containing class "c" pixels.
            msk_class = new_msk[msk_ravel == c]

            # Random permutation of class "c" pixels.
            perm = np.random.permutation(msk_class.shape[0])
            if type(sparsity) is float or type(sparsity) is int:
                sparsity_num = round(sparsity)
            else:
                sparsity_num = np.random.randint(low=1, high=len(perm))
            msk_class[perm[: min(sparsity_num, len(perm))]] = c

            # Merging sparse masks.
            new_msk[msk_ravel == c] = msk_class

        # Reshaping linearized sparse mask to the original 2 dimensions.
        new_msk = new_msk.reshape(msk.shape)

        np.random.seed(None)

        return new_msk

    @staticmethod
    def sparse_grid_old(
        msk: NDArray, sparsity: SparsityValue = "random", seed=0
    ) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:, :] = -1

        if type(sparsity) is float or type(sparsity) is int:
            spacing_value = int(sparsity)
        else:
            max_high = int(np.max(msk.shape) / 2)
            spacing_value = np.random.randint(low=1, high=max_high)
        spacing = (spacing_value, spacing_value)

        starting = (np.random.randint(spacing[0]), np.random.randint(spacing[1]))

        new_msk[starting[0] :: spacing[0], starting[1] :: spacing[1]] = msk[
            starting[0] :: spacing[0], starting[1] :: spacing[1]
        ]

        np.random.seed(None)

        return new_msk


class RimOneDataset(OpticDiscCupDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        return get_rim_one_data_path()

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        return read_image_mask(img_path, msk_path)


class DrishtiDataset(OpticDiscCupDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        return get_drishti_data_path()

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        return read_image_mask(img_path, msk_path)


class RimOneSimpleDataset(SimpleDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        return get_rim_one_data_path()

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        return read_image_mask(img_path, msk_path)


class DrishtiSimpleDataset(SimpleDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        return get_drishti_data_path()

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        return read_image_mask(img_path, msk_path)
