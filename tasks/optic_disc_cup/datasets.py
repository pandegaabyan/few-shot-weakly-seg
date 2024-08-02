import os
from abc import ABC

import numpy as np
from numpy.typing import NDArray
from skimage import io

from data.base_dataset import BaseDataset
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import DataPathList, SparsityMode, SparsityValue


def get_rim_one_data_path() -> DataPathList:
    data_path = "../data/RIM-ONE DL"
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
    data_path = "../data/DRISHTI-GS"
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
    msk = io.imread(msk_path, as_gray=True).astype(np.int8)

    return img, msk


class OpticDiscCupDataset(FewSparseDataset, ABC):
    def set_class_labels(self) -> dict[int, str]:
        return {0: "background", 1: "optic_disc", 2: "optic_cup"}

    def set_additional_sparse_mode(self) -> list[SparsityMode]:
        return []

    def get_additional_sparse_mask(
        self,
        sparsity_mode: SparsityMode,
        msk: NDArray,
        img: NDArray | None = None,
        sparsity_value: SparsityValue = "random",
        seed=0,
    ) -> NDArray:
        return msk


class OpticDiscCupSimpleDataset(SimpleDataset, ABC):
    def set_class_labels(self) -> dict[int, str]:
        return {0: "background", 1: "optic_disc", 2: "optic_cup"}


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


class RimOneSimpleDataset(OpticDiscCupSimpleDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        return get_rim_one_data_path()

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        return read_image_mask(img_path, msk_path)


class DrishtiSimpleDataset(OpticDiscCupSimpleDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        return get_drishti_data_path()

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        return read_image_mask(img_path, msk_path)
