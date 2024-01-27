from typing import Type

from torch.utils.data import DataLoader
from typing_extensions import TypedDict

from config.config_type import DataConfig
from data.simple_dataset import SimpleDataset
from data.types import SimpleDatasetKeywordArgs


class SimpleDatasetLoaderItem(TypedDict):
    train: DataLoader
    test: DataLoader
    val: DataLoader


def get_simple_dataset_loader(
    data_config: DataConfig,
    dataset_class: Type[SimpleDataset],
    dataset_kwargs: SimpleDatasetKeywordArgs,
    test_dataset_class: Type[SimpleDataset] | None = None,
    test_dataset_kwargs: SimpleDatasetKeywordArgs | None = None,
    pin_memory: bool = False,
) -> SimpleDatasetLoaderItem:
    train_dataset = dataset_class(
        "train", data_config["num_classes"], data_config["resize_to"], **dataset_kwargs
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_dataset = dataset_class(
        "val", data_config["num_classes"], data_config["resize_to"], **dataset_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        shuffle=True,
        pin_memory=pin_memory,
    )

    if test_dataset_class is None or test_dataset_kwargs is None:
        test_dataset_class = dataset_class
        test_dataset_kwargs = dataset_kwargs

    test_dataset = test_dataset_class(
        "test",
        data_config["num_classes"],
        data_config["resize_to"],
        **test_dataset_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=data_config["num_workers"],
        shuffle=False,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
