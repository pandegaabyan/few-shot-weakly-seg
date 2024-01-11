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


def get_simple_dataset_loader(dataset_class: Type[SimpleDataset], dataset_kwargs: SimpleDatasetKeywordArgs,
                              data_config: DataConfig, pin_memory: bool = False) -> SimpleDatasetLoaderItem:
    train_dataset = dataset_class(
        'train',
        data_config['num_classes'],
        data_config['resize_to'],
        **dataset_kwargs
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        shuffle=True,
        pin_memory=pin_memory
    )

    val_dataset = dataset_class(
        'val',
        data_config['num_classes'],
        data_config['resize_to'],
        **dataset_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        shuffle=True,
        pin_memory=pin_memory
    )

    test_dataset = dataset_class(
        'test',
        data_config['num_classes'],
        data_config['resize_to'],
        **dataset_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=data_config['num_workers'],
        shuffle=False,
        pin_memory=pin_memory
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
