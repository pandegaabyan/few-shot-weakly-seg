from typing import TypedDict, Literal, Union

DatasetModes = Literal["train", "test", "meta_train", "meta_test", "tune_train", "tune_test"]

SparsityModes = Union[Literal["point", "grid", "contour", "skeleton", "region"], Literal["dense", "random"], str]

SparsityValue = Union[float, Literal["random"]]

SparsityDict = dict[str, list[SparsityValue]]


class FewSparseDatasetKeywordArgs(TypedDict, total=False):
    num_shots: int
    split_seed: int
    split_test_size: float
    sparsity_mode: SparsityModes
    sparsity_value: SparsityValue
    sparsity_params: dict
