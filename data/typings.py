from typing import Literal, NamedTuple, TypedDict, TypeVar, Union

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
    file_name: str
    dataset_name: str


class SupportDataTuple(NamedTuple):
    images: Tensor
    masks: Tensor
    file_names: list[str]
    sparsity_mode: Union[SparsityMode, list[SparsityMode]]
    sparsity_value: Union[SparsityValue, list[SparsityValue]]


class QueryDataTuple(NamedTuple):
    images: Tensor
    masks: Tensor
    file_names: list[str]


class FewSparseDataTuple(NamedTuple):
    support: SupportDataTuple
    query: QueryDataTuple
    dataset_name: str


class BaseDatasetKwargs(TypedDict, total=False):
    max_items: int | None
    seed: int | None
    split_val_size: float
    split_val_fold: int
    split_test_size: float
    split_test_fold: int
    cache_data: bool
    dataset_name: str | None


class SimpleDatasetKwargs(BaseDatasetKwargs):
    ...


class FewSparseDatasetKwargs(BaseDatasetKwargs):
    shot_options: ShotOptions
    sparsity_options: SparsityOptions
    sparsity_params: dict | None
    homogen_support_batch: bool
    query_batch_size: int
    split_query_size: float
    split_query_fold: int
    num_iterations: int | float
