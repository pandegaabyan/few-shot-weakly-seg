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

NUM_CLASSES = 33


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


class TeethBaseDataset(BaseDataset, ABC):
    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        img = io.imread(img_path, as_gray=False)
        msk = io.imread(msk_path, as_gray=True)
        msk = (msk // (255 // (NUM_CLASSES - 1))).astype(np.int8)
        return img, msk

    def set_class_labels(self) -> dict[int, str]:
        return {
            0: "background",
            1: "up_right_3rd_molar",
            2: "up_right_2nd_molar",
            3: "up_right_1st_molar",
            4: "up_right_2nd_premolar",
            5: "up_right_1st_premolar",
            6: "up_right_canine",
            7: "up_right_lateral_incisor",
            8: "up_right_central_incisor",
            9: "up_left_central_incisor",
            10: "up_left_lateral_incisor",
            11: "up_left_canine",
            12: "up_left_1st_premolar",
            13: "up_left_2nd_premolar",
            14: "up_left_1st_molar",
            15: "up_left_2nd_molar",
            16: "up_left_3rd_molar",
            17: "low_left_3rd_molar",
            18: "low_left_2nd_molar",
            19: "low_left_1st_molar",
            20: "low_left_2nd_premolar",
            21: "low_left_1st_premolar",
            22: "low_left_canine",
            23: "low_left_lateral_incisor",
            24: "low_left_central_incisor",
            25: "low_right_central_incisor",
            26: "low_right_lateral_incisor",
            27: "low_right_canine",
            28: "low_right_1st_premolar",
            29: "low_right_2nd_premolar",
            30: "low_right_1st_molar",
            31: "low_right_2nd_molar",
            32: "low_right_3rd_molar",
        }


class TeethFSDataset(TeethBaseDataset, FewSparseDataset, ABC):
    @staticmethod
    def sparse_region(*args, **kwargs) -> NDArray:
        raise NotImplementedError

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


class TeethSimpleDataset(TeethBaseDataset, SimpleDataset, ABC): ...


def create_dataset_classes(
    data_dir,
) -> tuple[Type[SimpleDataset], Type[FewSparseDataset]]:
    data_path = "../data/" + data_dir

    class SimpleDataset(TeethSimpleDataset):
        def get_all_data_path(self) -> DataPathList:
            return get_all_data_path(data_path)

    class FewSparseDataset(TeethFSDataset):
        def get_all_data_path(self) -> DataPathList:
            return get_all_data_path(data_path)

    return SimpleDataset, FewSparseDataset


HITLSimpleDataset, HITLFSDataset = create_dataset_classes("HITL")
AdnanUmerSimpleDataset, AdnanUmerFSDataset = create_dataset_classes("Adnan-Umer")
DualLabeledSimpleDataset, DualLabeledFSDataset = create_dataset_classes("Dual-Labeled")
TuftsSimpleDataset, TuftsFSDataset = create_dataset_classes("Tufts")
UFBA425SimpleDataset, UFBA425FSDataset = create_dataset_classes("UFBA-425")

teeth_sparsity_params_512: dict = {
    "point_dot_size": 9,
    "grid_spacing": 20,
    "grid_dot_size": 6,
    "contour_radius_dist": 5,
    "contour_radius_thick": 2,
    "skeleton_radius_thick": 3,
}

teeth_sparsity_params_256: dict = {
    "point_dot_size": 4,
    "grid_spacing": 10,
    "grid_dot_size": 3,
    "contour_radius_dist": 2,
    "contour_radius_thick": 1,
    "skeleton_radius_thick": 1,
}
