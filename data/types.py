from typing import TypedDict, Literal, Union

from torch import Tensor

DatasetModes = Literal["train", "test", "meta_train", "meta_test", "tune_train", "tune_test"]

SimpleDatasetModes = Literal["train", "test", "val"]

SparsityModes = Union[Literal["point", "grid", "contour", "skeleton", "region"], Literal["dense", "random"], str]

SparsityValue = Union[float, tuple[float, float], int, tuple[int, int], Literal["random"]]

SparsityDict = dict[str, list[SparsityValue]]

TensorDataItem = tuple[Tensor, Tensor, Tensor, str]

SimpleTensorDataItem = tuple[Tensor, Tensor, str]


class FewSparseDatasetKeywordArgs(TypedDict, total=False):
    num_shots: int
    split_seed: int
    split_test_size: float
    sparsity_mode: SparsityModes
    sparsity_value: SparsityValue
    sparsity_params: dict


class SimpleDatasetKeywordArgs(TypedDict, total=False):
    split_seed: int
    split_val_size: float
    split_test_size: float
