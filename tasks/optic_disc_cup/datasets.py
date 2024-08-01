import os

import numpy as np
from numpy.typing import NDArray
from skimage import io

from data.few_sparse_dataset import FewSparseDataset


class RimOneDataset(FewSparseDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        data_path = "../data/RIM-ONE DL"
        img_dir = "/images/"
        msk_dir = "/masks/"
        img_files = os.listdir(data_path + img_dir)
        img_files_no_ext = list(map(self.get_filename_from_path, img_files))
        msk_files = os.listdir(data_path + msk_dir)

        all_data_path = []
        for msk_file in msk_files:
            msk_file_no_ext = self.get_filename_from_path(msk_file)
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

    @staticmethod
    def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True).astype(np.int8)

        return img, msk


class DrishtiDataset(FewSparseDataset):
    def get_all_data_path(self) -> list[tuple[str, str]]:
        data_path = "../data/DRISHTI-GS"
        img_dir = "/images/"
        msk_dir = "/masks/"
        img_files = os.listdir(data_path + img_dir)
        img_files_no_ext = list(map(self.get_filename_from_path, img_files))
        msk_files = os.listdir(data_path + msk_dir)

        all_data_path = []
        for msk_file in msk_files:
            msk_file_no_ext = self.get_filename_from_path(msk_file)
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

    @staticmethod
    def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True).astype(np.int8)

        return img, msk
