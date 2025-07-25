import os
from abc import ABC
from typing import Type

import numpy as np
from numpy.typing import NDArray
from skimage import io

from data.base_dataset import BaseDataset
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import DataPathList, SparsityMode, SparsityValue


def get_all_data_path(dir: str) -> DataPathList:
    img_dir = "/input/"
    msk_dir = "/mask/"
    img_files = sorted(os.listdir(dir + img_dir))
    msk_files = sorted(os.listdir(dir + msk_dir))

    all_data_path = []
    for img_file, msk_file in zip(img_files, msk_files):
        all_data_path.append(
            (
                dir + img_dir + img_file,
                dir + msk_dir + msk_file,
            )
        )

    return all_data_path


class SkinLesionBaseDataset(BaseDataset, ABC):
    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True)
        msk = (msk / 255).astype(np.int8)
        return img, msk

    def set_class_labels(self) -> dict[int, str]:
        return {0: "background", 1: "lesion"}


class SkinLesionFSDataset(SkinLesionBaseDataset, FewSparseDataset, ABC):
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


class SkinLesionSimpleDataset(SkinLesionBaseDataset, SimpleDataset, ABC): ...


def create_dataset_classes(
    data_dir,
) -> tuple[Type[SimpleDataset], Type[FewSparseDataset]]:
    data_path = "../data/" + data_dir

    class SimpleDataset(SkinLesionSimpleDataset):
        def get_all_data_path(self) -> DataPathList:
            return get_all_data_path(data_path)

    class FewSparseDataset(SkinLesionFSDataset):
        def get_all_data_path(self) -> DataPathList:
            return get_all_data_path(data_path)

    return SimpleDataset, FewSparseDataset


ISIC16SimpleDataset, ISIC16FSDataset = create_dataset_classes("ISIC16")
ISIC17SimpleDataset, ISIC17FSDataset = create_dataset_classes("ISIC17")
ISIC18SimpleDataset, ISIC18FSDataset = create_dataset_classes("ISIC18")
PH2SimpleDataset, PH2FSDataset = create_dataset_classes("PH2")
