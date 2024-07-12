from typing import Literal, Union

from typing_extensions import NotRequired, TypedDict

RunMode = Literal["fit-test", "fit", "test", "study"]

OptunaSampler = Literal["random", "tpe", "cmaes", "qmc", "gp"]
OptunaPruner = Literal["none", "median", "percentile", "asha", "hyperband", "threshold"]

# SEPARATE LOG RELATED CONFIG ?


class DataConfig(TypedDict):
    num_classes: int
    num_channels: int
    num_workers: int
    batch_size: int
    resize_to: tuple[int, int]


class LearnConfig(TypedDict):
    num_epochs: int
    exp_name: str
    run_name: str
    dummy: NotRequired[bool]
    val_freq: NotRequired[int]
    model_onnx: NotRequired[bool]
    tensorboard_graph: NotRequired[bool]
    manual_optim: NotRequired[bool]
    ref_ckpt_path: NotRequired[str | None]
    optuna_study_name: NotRequired[str]


class OptimizerConfig(TypedDict, total=False):
    lr: float
    lr_bias: float
    weight_decay: float
    weight_decay_bias: float
    betas: tuple[float, float]


class SchedulerConfig(TypedDict, total=False):
    step_size: int
    gamma: float


class CallbacksConfig(TypedDict, total=False):
    progress: bool
    progress_leave: bool
    monitor: str
    monitor_mode: Literal["min", "max"]
    ckpt_last: bool
    ckpt_top_k: int
    stop_patience: int
    stop_min_delta: float
    stop_threshold: float | None


class WandbConfig(TypedDict):
    run_id: str
    tags: NotRequired[list[str]]
    job_type: NotRequired[str]
    watch_model: NotRequired[bool]
    push_table_freq: NotRequired[int]
    save_train_preds: NotRequired[int]
    save_val_preds: NotRequired[int]
    save_test_preds: NotRequired[int]


class SimpleLearnerConfig(TypedDict):
    ...


class MetaLearnerConfig(TypedDict):
    ...


class WeaselConfig(TypedDict):
    use_first_order: bool
    update_param_step_size: float
    tune_epochs: int
    tune_val_freq: int


class ProtoSegConfig(TypedDict):
    embedding_size: int


class GuidedNetsConfig(TypedDict):
    embedding_size: int


class ConfigBase(TypedDict):
    data: DataConfig
    learn: LearnConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    callbacks: CallbacksConfig
    wandb: NotRequired[WandbConfig]


class ConfigSimpleLearner(ConfigBase):
    simple_learner: SimpleLearnerConfig


class ConfigMetaLearner(ConfigBase):
    meta_learner: MetaLearnerConfig


class ConfigWeasel(ConfigMetaLearner):
    weasel: WeaselConfig


class ConfigProtoSeg(ConfigMetaLearner):
    protoseg: ProtoSegConfig


class ConfigGuidedNets(ConfigMetaLearner):
    guidednets: GuidedNetsConfig


ConfigUnion = Union[
    ConfigBase,
    ConfigSimpleLearner,
    ConfigMetaLearner,
    ConfigWeasel,
    ConfigProtoSeg,
    ConfigGuidedNets,
]
