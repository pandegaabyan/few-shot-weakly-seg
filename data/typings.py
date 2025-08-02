from typing import Literal, NamedTuple, TypedDict, TypeVar, Union

from albumentations import BaseCompose, BasicTransform
from numpy.typing import NDArray
from torch import Tensor

T = TypeVar("T")

DatasetModes = Literal["train", "test", "val"]

DataPathList = list[tuple[str, str]]

SparsityMode = Union[
    Literal["point", "grid", "contour", "skeleton", "region"],
    Literal["dense", "random"],
    str,
]

SparsityValue = Union[float, int, Literal["random"]]

SparsityValueOptions = Union[
    float,
    list[float],
    tuple[float, float],
    int,
    list[int],
    tuple[int, int],
    Literal["random"],
]

SparsityTuple = tuple[SparsityMode, SparsityValue]

SparsityOptions = list[tuple[SparsityMode, SparsityValueOptions]]

ShotOptions = Union[int, list[int], tuple[int, int], Literal["random", "all"]]


class BaseDataTuple(NamedTuple):
    image: NDArray
    mask: NDArray
    file_name: str


class SimpleDataTuple(NamedTuple):
    image: Tensor
    mask: Tensor
    index: int
    dataset_name: str


class SupportDataTuple(NamedTuple):
    images: Tensor
    masks: Tensor
    indices: list[int]
    sparsity_mode: Union[SparsityMode, list[SparsityMode]]
    sparsity_value: Union[SparsityValue, list[SparsityValue]]


class QueryDataTuple(NamedTuple):
    images: Tensor
    masks: Tensor
    indices: list[int]


class FewSparseDataTuple(NamedTuple):
    support: SupportDataTuple
    query: QueryDataTuple
    dataset_name: str


class BaseDatasetKwargs(TypedDict, total=False):
    seed: int
    size: float | int
    split_val_size: float
    split_val_fold: int
    split_test_size: float
    split_test_fold: int
    cache_data: bool
    dataset_name: str | None
    transforms: BaseCompose | BasicTransform | None


class SimpleDatasetKwargs(BaseDatasetKwargs): ...


class FewSparseDatasetKwargs(BaseDatasetKwargs, total=False):
    shot_options: ShotOptions
    sparsity_options: SparsityOptions
    sparsity_params: dict | None
    support_query_data: Literal["split", "mixed", "mixed_replaced"]
    support_batch_mode: Literal["mixed", "homogen", "permutation", "full_permutation"]
    query_batch_size: int
    split_query_size: float
    split_query_fold: int
    num_iterations: int | float
