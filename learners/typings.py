import optuna
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch import Tensor, nn, optim
from typing_extensions import Any, Generic, Required, Type, TypedDict, TypeVar, Union

from config.config_type import (
    ConfigBase,
    ConfigGuidedNets,
    ConfigMetaLearner,
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigWeasel,
)
from data.base_dataset import BaseDataset
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import (
    BaseDatasetKwargs,
    FewSparseDatasetKwargs,
    SimpleDatasetKwargs,
)
from learners.metrics import BaseMetric

ConfigType = TypeVar("ConfigType", bound=ConfigBase)
ConfigTypeMeta = TypeVar("ConfigTypeMeta", bound=ConfigMetaLearner)
DatasetClass = TypeVar("DatasetClass", bound=BaseDataset)
DatasetKwargs = TypeVar("DatasetKwargs", bound=BaseDatasetKwargs)

Primitives = Union[bool, str, int, float, None]
ListPrimitives = Union[
    list[bool],
    list[str],
    list[int],
    list[float],
]

Loss = nn.Module
Metric = BaseMetric
Optimizer = optim.Optimizer
Scheduler = LRSchedulerPLType


class DatasetLists(TypedDict, Generic[DatasetClass, DatasetKwargs], total=False):
    dataset_list: Required[list[tuple[Type[DatasetClass], DatasetKwargs]]]
    val_dataset_list: list[tuple[Type[DatasetClass], DatasetKwargs]]
    test_dataset_list: list[tuple[Type[DatasetClass], DatasetKwargs]]


class BaseLearnerKwargs(
    TypedDict, Generic[ConfigType, DatasetClass, DatasetKwargs], total=False
):
    config: Required[ConfigType]
    dataset_list: Required[list[tuple[Type[DatasetClass], DatasetKwargs]]]
    val_dataset_list: list[tuple[Type[DatasetClass], DatasetKwargs]]
    test_dataset_list: list[tuple[Type[DatasetClass], DatasetKwargs]]
    loss: tuple[Type[Loss], dict[str, Any]]
    metric: tuple[Type[Metric], dict[str, Any]]
    optuna_trial: optuna.Trial | None


SimpleLearnerKwargs = BaseLearnerKwargs[
    ConfigSimpleLearner, SimpleDataset, SimpleDatasetKwargs
]


class MetaLearnerKwargs(
    Generic[ConfigTypeMeta],
    BaseLearnerKwargs[ConfigTypeMeta, FewSparseDataset, FewSparseDatasetKwargs],
): ...


WeaselLearnerKwargs = MetaLearnerKwargs[ConfigWeasel]

ProtoSegLearnerKwargs = MetaLearnerKwargs[ConfigProtoSeg]

GuidedNetsLearnerKwargs = MetaLearnerKwargs[ConfigGuidedNets]


SimpleDataBatchTuple = tuple[Tensor, Tensor, list[str], list[str]]

PredictionDataDict = dict[
    str, list[tuple[Tensor | None, Tensor, Tensor, list[tuple[str, Primitives]]]]
]
