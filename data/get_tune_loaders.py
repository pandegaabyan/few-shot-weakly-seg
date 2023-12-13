from typing import Type, TypedDict

from data.few_sparse_dataset import FewSparseDataset, FewSparseDatasetKeywordArgs, SparsityModesNoRandom
from torch.utils.data import DataLoader


class TuneLoaderListItem(TypedDict):
    n_shots: int
    sparsity: float
    train: DataLoader
    test: DataLoader


class TuneLoaderDict(TypedDict):
    point: list[TuneLoaderListItem]
    grid: list[TuneLoaderListItem]
    contour: list[TuneLoaderListItem]
    skeleton: list[TuneLoaderListItem]
    region: list[TuneLoaderListItem]
    dense: list[TuneLoaderListItem]


def get_tune_loaders(dataset_class: Type[FewSparseDataset],
                     dataset_kwargs: FewSparseDatasetKeywordArgs,
                     num_classes: int,
                     resize_to: tuple[int, int],
                     shots: list[int],
                     point: list[float],
                     contour: list[float],
                     grid: list[float],
                     region: list[float],
                     skeleton: list[float],
                     batch_size: int = 1,
                     num_workers: int = 0
                     ) -> TuneLoaderDict:
    sparsity_modes: list[SparsityModesNoRandom] = ['point', 'grid', 'contour', 'skeleton', 'region', 'dense']
    loader_dict = {}

    dataset_kwargs.pop('sparsity_mode')
    dataset_kwargs.pop('sparsity_value')

    for sparsity_mode in sparsity_modes:
        loader_list = []

        if sparsity_mode == 'point':
            sparsity_values = point
        elif sparsity_mode == 'grid':
            sparsity_values = grid
        elif sparsity_mode == 'contour':
            sparsity_values = contour
        elif sparsity_mode == 'skeleton':
            sparsity_values = skeleton
        elif sparsity_mode == 'region':
            sparsity_values = region
        else:
            sparsity_values = [-1]

        for n_shots in shots:
            for sparsity_value in sparsity_values:
                tune_train_set = dataset_class('tune_train',
                                               num_classes,
                                               n_shots,
                                               resize_to,
                                               sparsity_mode=sparsity_mode,
                                               sparsity_value=sparsity_value,
                                               **dataset_kwargs)
                tune_train_loader = DataLoader(tune_train_set,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

                tune_test_set = dataset_class('tune_test',
                                              num_classes,
                                              n_shots,
                                              resize_to,
                                              sparsity_mode='dense',
                                              **dataset_kwargs)
                tune_test_loader = DataLoader(tune_test_set,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              shuffle=False)

                loader_list.append({
                    'n_shots': n_shots,
                    'sparsity': sparsity_value,
                    'train': tune_train_loader,
                    'test': tune_test_loader
                })

        loader_dict[sparsity_mode] = loader_list

    return loader_dict
