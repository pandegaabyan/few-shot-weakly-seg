from abc import ABC

import numpy as np
from numpy.typing import NDArray

from data.base_dataset import BaseDataset
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import DataPathList


class DummyBaseDataset(BaseDataset, ABC):
    def set_class_labels(self) -> dict[int, str]:
        return {0: "background", 1: "optic_disc", 2: "optic_cup"}

    def get_all_data_path(self) -> DataPathList:
        return [
            (f"path/to/image/img{index}.png", f"path/to/mask/msk{index}.png")
            for index in range(400)
        ]

    def read_image_mask(self, img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        size = np.random.randint(300, 500)
        num_classes = 3
        img = np.random.randint(0, 255, (size, size, 3)).astype(np.uint8)
        # msk = np.random.randint(0, num_classes, (size, size)).astype(np.int8)
        msk = np.zeros((size, size), dtype=np.int8)
        for i in range(num_classes):
            step = size // num_classes
            msk[i * step : (i + 1) * step, :] = i
        return img, msk


class DummyDataset(DummyBaseDataset, FewSparseDataset):
    def set_additional_sparse_mode(self):
        return []

    def get_additional_sparse_mask(
        self, sparsity_mode, msk, img=None, sparsity_value="random", seed=0
    ) -> NDArray:
        return msk


class DummySimpleDataset(DummyBaseDataset, SimpleDataset): ...
