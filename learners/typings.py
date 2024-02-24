from typing import Type, TypedDict, TypeVar

from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch import Tensor, optim
from typing_extensions import Required

from config.config_type import ConfigBase, ConfigSimpleLearner
from data.base_dataset import BaseDataset
from data.simple_dataset import SimpleDataset
from data.typings import BaseDatasetKwargs, SimpleDatasetKwargs
from learners.losses import CustomLoss
from learners.metrics import CustomMetric

ConfigType = TypeVar("ConfigType", bound=ConfigBase)

DatasetClass = TypeVar("DatasetClass", bound=BaseDataset)

DatasetKwargs = TypeVar("DatasetKwargs", bound=BaseDatasetKwargs)

Optimizer = optim.Optimizer

Scheduler = LRSchedulerPLType


class SimpleLearnerKwargs(TypedDict, total=False):
    config: Required[ConfigSimpleLearner]
    dataset_list: Required[list[tuple[Type[SimpleDataset], SimpleDatasetKwargs]]]
    val_dataset_list: list[tuple[Type[SimpleDataset], SimpleDatasetKwargs]]
    test_dataset_list: list[tuple[Type[SimpleDataset], SimpleDatasetKwargs]]
    loss: CustomLoss | None
    metric: CustomMetric | None
    resume: bool
    force_clear_dir: bool


SimpleDataBatchTuple = tuple[Tensor, Tensor, list[str], list[str]]
