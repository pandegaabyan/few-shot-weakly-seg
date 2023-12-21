import os
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from skimage import io

from data.few_sparse_dataset import FewSparseDataset
from data.types import SparsityModes, SparsityValue


class OpticDiscCupDataset(FewSparseDataset, ABC):
    @abstractmethod
    def get_all_data_path(self) -> list[tuple[str, str]]:
        pass

    @staticmethod
    @abstractmethod
    def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        pass

    def set_additional_sparse_mode(self) -> list[SparsityModes]:
        return []

    def get_additional_sparse_mask(self, sparsity_mode: SparsityModes, msk: NDArray, img: NDArray = None,
                                   sparsity_value: SparsityValue = 'random', seed=0) -> NDArray:
        pass


class RimOneDataset(OpticDiscCupDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        data_path = "../Data/RIM-ONE"
        img_dir = "/images/"
        msk_dir = "/masks/"
        img_files = os.listdir(data_path + img_dir)
        img_files_no_ext = list(map(self.filename_from_path, img_files))
        msk_files = os.listdir(data_path + msk_dir)

        all_data_path = []
        for msk_file in msk_files:
            msk_file_no_ext = self.filename_from_path(msk_file)
            try:
                img_index = img_files_no_ext.index(msk_file_no_ext)
                all_data_path.append((
                    data_path + img_dir + img_files[img_index],
                    data_path + msk_dir + msk_file,
                ))
            except ValueError:
                continue

        return all_data_path

    @staticmethod
    def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True)
        msk = (msk * 255).astype(np.int8)

        return img, msk


class DrishtiDataset(OpticDiscCupDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        data_path = "../Data/DRISHTI-GS"
        img_dir = "/images/"
        msk_dir = "/masks/"
        img_files = os.listdir(data_path + img_dir)
        img_files_no_ext = list(map(self.filename_from_path, img_files))
        msk_files = os.listdir(data_path + msk_dir)

        all_data_path = []
        for msk_file in msk_files:
            msk_file_no_ext = self.filename_from_path(msk_file)
            try:
                img_index = img_files_no_ext.index(msk_file_no_ext)
                all_data_path.append((
                    data_path + img_dir + img_files[img_index],
                    data_path + msk_dir + msk_file,
                ))
            except ValueError:
                continue

        return all_data_path

    @staticmethod
    def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True)
        msk = (msk * 255).astype(np.int8)

        return img, msk
