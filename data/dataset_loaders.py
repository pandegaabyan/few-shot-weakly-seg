import copy
from typing import Type, Literal

from torch.utils.data import DataLoader
from typing_extensions import NotRequired, TypedDict

from config.config_type import DataConfig, DataTuneConfig
from data.few_sparse_dataset import FewSparseDataset
from data.types import DatasetModes, SparsityModes, SparsityValue, FewSparseDatasetKeywordArgs

DatasetModesSimple = Literal["", "meta", "tune"]


class DatasetLoaderItem(TypedDict):
    n_shots: int
    sparsity_mode: SparsityModes
    sparsity_value: SparsityValue
    train: DataLoader
    test: DataLoader


class DatasetLoaderParam(TypedDict):
    dataset_class: Type[FewSparseDataset]
    dataset_kwargs: FewSparseDatasetKeywordArgs
    mode: DatasetModesSimple
    num_classes: int
    resize_to: tuple[int, int]
    pin_memory: NotRequired[bool]
    num_workers: NotRequired[int]
    train_batch_size: NotRequired[int]
    test_batch_size: NotRequired[int]


class DatasetLoaderParamSimple(TypedDict):
    dataset_class: Type[FewSparseDataset]
    dataset_kwargs: FewSparseDatasetKeywordArgs


class DatasetLoaderParamComplement(TypedDict):
    mode: DatasetModesSimple
    num_classes: int
    resize_to: tuple[int, int]
    pin_memory: NotRequired[bool]
    num_workers: NotRequired[int]
    train_batch_size: NotRequired[int]
    test_batch_size: NotRequired[int]


def get_dataset_loaders(param_list: list[DatasetLoaderParam]) -> list[DatasetLoaderItem]:
    dataset_loaders: list[DatasetLoaderItem] = []

    for param in param_list:
        dataset_class = param['dataset_class']
        kwargs = copy.deepcopy(param['dataset_kwargs'])

        train_mode: DatasetModes = param['mode'] + '_train' if param['mode'] != '' else 'train'  # type: ignore
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
            shuffle=True,
            pin_memory=param.get('pin_memory', False)
        )

        test_mode: DatasetModes = param['mode'] + '_test' if param['mode'] != '' else 'test'  # type: ignore
        kwargs.pop('sparsity_mode')
        kwargs.pop('sparsity_value')
        test_dataset = dataset_class(
            test_mode,
            param['num_classes'],
            param['resize_to'],
            sparsity_mode="dense",
            **kwargs  # type: ignore
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=param.get('test_batch_size', 1),
            num_workers=param.get('num_workers', 0),
            shuffle=False,
            pin_memory=param.get('pin_memory', False)
        )

        dataset_loaders.append({
            'n_shots': param['dataset_kwargs'].get('num_shots', -1),
            'sparsity_mode': param['dataset_kwargs'].get('sparsity_mode', 'random'),
            'sparsity_value': param['dataset_kwargs'].get('sparsity_value', 'random'),
            'train': train_loader,
            'test': test_loader
        })

    return dataset_loaders


def get_meta_loaders(param_list: list[DatasetLoaderParamSimple], data_config: DataConfig,
                     pin_memory: bool = False) -> list[DatasetLoaderItem]:
    new_param_list: list[DatasetLoaderParam] = []
    complement_param: DatasetLoaderParamComplement = {
        'mode': 'meta',
        'num_classes': data_config['num_classes'],
        'resize_to': data_config['resize_to'],
        'pin_memory': pin_memory,
        'num_workers': data_config['num_workers'],
        'train_batch_size': data_config['batch_size'],
        'test_batch_size': data_config['batch_size'],
    }
    for simple_param in param_list:
        full_param: DatasetLoaderParam = {**simple_param, **complement_param}  # type: ignore
        new_param_list.append(full_param)

    return get_dataset_loaders(new_param_list)


def get_tune_loaders(param: DatasetLoaderParamSimple, data_config: DataConfig,
                     data_tune_config: DataTuneConfig, pin_memory: bool = False) -> list[DatasetLoaderItem]:
    new_param_list: list[DatasetLoaderParam] = []
    for shot in data_tune_config['shot_list']:
        for sparsity_mode, sparsity_values in data_tune_config['sparsity_dict'].items():
            for sparsity_value in sparsity_values:
                simple_param = copy.deepcopy(param)
                simple_param['dataset_kwargs']['num_shots'] = shot
                simple_param['dataset_kwargs']['sparsity_mode'] = sparsity_mode
                simple_param['dataset_kwargs']['sparsity_value'] = sparsity_value
                complement_param: DatasetLoaderParamComplement = {
                    'mode': 'tune',
                    'num_classes': data_config['num_classes'],
                    'resize_to': data_config['resize_to'],
                    'pin_memory': pin_memory,
                    'num_workers': data_config['num_workers'],
                    'train_batch_size': data_config['batch_size'],
                    'test_batch_size': 1,
                }
                full_param: DatasetLoaderParam = {**simple_param, **complement_param}  # type: ignore
                new_param_list.append(full_param)

    return get_dataset_loaders(new_param_list)
