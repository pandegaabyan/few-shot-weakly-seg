import copy
from typing import Type, Literal
from typing_extensions import NotRequired, TypedDict

from torch.utils.data import DataLoader

from data.few_sparse_dataset import FewSparseDataset
from data.types import SparsityDict, DatasetModes, SparsityModes, SparsityValue, FewSparseDatasetKeywordArgs


class DatasetLoaderItem(TypedDict):
    n_shots: int
    max_iterations: int | None
    sparsity_mode: SparsityModes
    sparsity_value: SparsityValue
    train: DataLoader
    test: DataLoader


class DatasetLoaderParam(TypedDict):
    dataset_class: Type[FewSparseDataset]
    dataset_kwargs: FewSparseDatasetKeywordArgs
    mode: Literal["", "meta", "tune"]
    num_classes: int
    resize_to: tuple[int, int]
    num_workers: NotRequired[int]
    train_batch_size: NotRequired[int]
    test_batch_size: NotRequired[int]
    max_iterations: NotRequired[int]


def get_dataset_loaders(param_list: list[DatasetLoaderParam]) -> list[DatasetLoaderItem]:
    dataset_loaders: list[DatasetLoaderItem] = []

    for param in param_list:
        dataset_class = param['dataset_class']
        kwargs = copy.deepcopy(param['dataset_kwargs'])

        train_mode: DatasetModes = param['mode'] + '_train' if param['mode'] != '' else 'train' # type: ignore
        train_dataset = dataset_class(
            train_mode,
            param['num_classes'],
            param['resize_to'],
            **kwargs
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=param.get('train_batch_size', 1),
            num_workers=param.get('num_workers', 0),
            shuffle=True
        )

        test_mode: DatasetModes = param['mode'] + '_test' if param['mode'] != '' else 'test' # type: ignore
        kwargs.pop('sparsity_mode')
        kwargs.pop('sparsity_value')
        test_dataset = dataset_class(
            test_mode,
            param['num_classes'],
            param['resize_to'],
            sparsity_mode="dense",
            **kwargs # type: ignore
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=param.get('test_batch_size', 1),
            num_workers=param.get('num_workers', 0),
            shuffle=False
        )

        dataset_loaders.append({
            'n_shots': param['dataset_kwargs'].get('num_shots', -1),
            'max_iterations': param.get('max_iterations', None),
            'sparsity_mode': param['dataset_kwargs'].get('sparsity_mode', 'random'),
            'sparsity_value': param['dataset_kwargs'].get('sparsity_value', 'random'),
            'train': train_loader,
            'test': test_loader
        })

    return dataset_loaders


def get_tune_loaders(param: DatasetLoaderParam, shots: list[int], sparsities: SparsityDict) -> list[DatasetLoaderItem]:
    param_list = []
    for shot in shots:
        for sparsity_mode, sparsity_values in sparsities.items():
            for sparsity_value in sparsity_values:
                new_param = copy.deepcopy(param)
                # noinspection PyTypedDict
                new_param['mode'] = 'tune'
                new_param['dataset_kwargs']['num_shots'] = shot
                new_param['dataset_kwargs']['sparsity_mode'] = sparsity_mode
                new_param['dataset_kwargs']['sparsity_value'] = sparsity_value
                param_list.append(new_param)

    return get_dataset_loaders(param_list)