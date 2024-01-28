from typing import Type, Union

from typing_extensions import NotRequired, TypedDict

from data.types import SparsityDict


class DataConfig(TypedDict):
    num_classes: int
    num_channels: int
    num_workers: int  # Number of workers on data loader.
    batch_size: int
    resize_to: tuple[int, int]


class LearnConfig(TypedDict):
    should_resume: bool
    use_gpu: bool
    num_epochs: int  # Number of epochs.
    exp_name: str


class LossConfig(TypedDict, total=False):
    type: str
    ignored_index: int


class OptimizerConfig(TypedDict, total=False):
    lr: float  # Learning rate.
    lr_bias: float
    weight_decay: float  # L2 penalty.
    weight_decay_bias: float
    betas: tuple[float, float]  # Momentum.


class SchedulerConfig(TypedDict, total=False):
    step_size: int
    gamma: float


class SimpleLearnerConfig(TypedDict):
    test_freq: NotRequired[int]


class MetaLearnerConfig(TypedDict):
    tune_freq: NotRequired[int]
    shot_list: list[int]  # Number of shots (i.e, total annotated samples)
    sparsity_dict: SparsityDict  # Sparsity of the annotations
    #   Point: number of labeled pixels in point annotation
    #   Grid: spacing between selected pixels in grid annotation
    #   Contour: density of the contours (1, is the complete contours)
    #   Skeleton: density of the skeletons (1, is the complete skeletons)
    #   Region: percentage of regions labeled (1, all \pure\ regions are labeled)


class WeaselConfig(TypedDict):
    use_first_order: bool  # First order approximation of MAML.
    update_param_step_size: float  # MAML inner loop step size.
    tune_epochs: int  # Number of epochs on the tuning phase.
    tune_test_freq: int  # Test each tune_test_freq epochs on the tuning phase.


class ProtoSegConfig(TypedDict):
    embedding_size: int


class GuidedNetsConfig(TypedDict):
    embedding_size: int


class ConfigBase(TypedDict):
    data: DataConfig
    learn: LearnConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


class ConfigSimpleLearner(TypedDict):
    data: DataConfig
    learn: LearnConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    simple_learner: SimpleLearnerConfig


class ConfigMetaLearner(TypedDict):
    data: DataConfig
    learn: LearnConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    meta_learner: MetaLearnerConfig


class ConfigWeasel(ConfigMetaLearner):
    weasel: WeaselConfig


class ConfigProtoSeg(ConfigMetaLearner):
    protoseg: ProtoSegConfig


class ConfigGuidedNets(ConfigMetaLearner):
    guidednets: GuidedNetsConfig


class ConfigAll(ConfigSimpleLearner, ConfigWeasel, ConfigProtoSeg, ConfigGuidedNets):
    ...


ConfigUnion = Union[
    ConfigBase,
    ConfigSimpleLearner,
    ConfigMetaLearner,
    ConfigWeasel,
    ConfigProtoSeg,
    ConfigGuidedNets,
    ConfigAll,
]


ConfigClassUnion = Union[
    Type[ConfigBase],
    Type[ConfigSimpleLearner],
    Type[ConfigMetaLearner],
    Type[ConfigWeasel],
    Type[ConfigProtoSeg],
    Type[ConfigGuidedNets],
    Type[ConfigAll],
]
