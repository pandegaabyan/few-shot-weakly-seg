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
    img_dir = "/images/"
    msk_dir = "/masks/"
    img_files = os.listdir(dir + img_dir)
    img_files_no_ext = [
        BaseDataset.filename_from_path(img_file) for img_file in img_files
    ]
    msk_files = os.listdir(dir + msk_dir)

    all_data_path = []
    for msk_file in msk_files:
        msk_file_no_ext = BaseDataset.filename_from_path(msk_file)
        try:
            img_index = img_files_no_ext.index(msk_file_no_ext)
            all_data_path.append(
                (
                    dir + img_dir + img_files[img_index],
                    dir + msk_dir + msk_file,
                )
            )
        except ValueError:
            continue

    return all_data_path


class OpticDiscCupBaseDataset(BaseDataset, ABC):
    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True).astype(np.int8)
        return img, msk

    def set_class_labels(self) -> dict[int, str]:
        return {0: "background", 1: "optic_disc", 2: "optic_cup"}


class OpticDiscCupFSDataset(OpticDiscCupBaseDataset, FewSparseDataset, ABC):
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


class OpticDiscCupSimpleDataset(OpticDiscCupBaseDataset, SimpleDataset, ABC): ...


def create_dataset_classes(
    data_path,
) -> tuple[Type[SimpleDataset], Type[FewSparseDataset]]:
    class SimpleDataset(OpticDiscCupSimpleDataset):
        def get_all_data_path(self) -> DataPathList:
            return get_all_data_path(data_path)

    class FewSparseDataset(OpticDiscCupFSDataset):
        def get_all_data_path(self) -> DataPathList:
            return get_all_data_path(data_path)

    return SimpleDataset, FewSparseDataset


RimOneSimpleDataset, RimOneFSDataset = create_dataset_classes("../data/RIM-ONE DL")
DrishtiSimpleDataset, DrishtiFSDataset = create_dataset_classes("../data/DRISHTI-GS")
OrigaSimpleDataset, OrigaFSDataset = create_dataset_classes("../data/ORIGA")
PapilaSimpleDataset, PapilaFSDataset = create_dataset_classes("../data/PAPILA")
