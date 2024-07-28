from typing import Type, TypedDict

from data.few_sparse_dataset import FewSparseDataset, FewSparseDatasetKeywordArgs


class MetaDatasetParam(TypedDict):
    dataset_class: Type[FewSparseDataset]
    num_classes: int
    resize_to: tuple[int, int]
    kwargs: FewSparseDatasetKeywordArgs


class MetaDatasets(TypedDict):
    train: list[FewSparseDataset]
    test: list[FewSparseDataset]


def get_meta_datasets(param_list: list[MetaDatasetParam]) -> MetaDatasets:
    meta_datasets: MetaDatasets = {"train": [], "test": []}

    for param in param_list:
        kwargs = param["kwargs"].copy()
        dataset_class = param["dataset_class"]
        train_dataset = dataset_class(
            "meta_train", param["num_classes"], -1, param["resize_to"], **kwargs
        )

        kwargs["sparsity_mode"] = "dense"
        test_dataset = dataset_class(
            "meta_test",
            param["num_classes"],
            -1,
            param["resize_to"],
            **kwargs,
        )

        meta_datasets["train"].append(train_dataset)
        meta_datasets["test"].append(test_dataset)

    return meta_datasets
