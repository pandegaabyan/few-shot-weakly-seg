from typing import Literal, Union

from typing_extensions import NotRequired, Required, TypedDict

RunMode = Literal["fit-test", "fit", "test", "study"]
LearnerType = Literal["simple", "weasel", "protoseg", "guidednets", None]
ProfilerType = Literal["simple", "advanced", "pytorch", "custom", None]


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
    cudnn_deterministic: NotRequired[bool | Literal["warn"]]
    cudnn_benchmark: NotRequired[bool]
    profiler: NotRequired[ProfilerType]
    manual_optim: NotRequired[bool]
    ref_ckpt: NotRequired[str | None]
    optuna_study: NotRequired[str | None]


class OptimizerConfig(TypedDict, total=False):
    lr: float
    lr_bias_mult: float
    weight_decay: float
    betas: tuple[float, float]


class SchedulerConfig(TypedDict, total=False):
    step_size: int
    gamma: float


class LogConfig(TypedDict, total=False):
    configuration: bool
    table: bool
    model_onnx: bool
    tensorboard_graph: bool


class CallbacksConfig(TypedDict, total=False):
    progress: bool
    monitor: str | None
    monitor_mode: Literal["min", "max"]
    ckpt_last: bool
    ckpt_top_k: int
    stop_patience: int
    stop_min_delta: float
    stop_threshold: float | None


class WandbConfig(TypedDict, total=False):
    run_id: Required[str]
    tags: list[str]
    job_type: str | None
    watch_model: bool
    save_model: bool
    push_table_freq: int | None
    save_train_preds: int
    save_val_preds: int
    save_test_preds: int


class SimpleLearnerConfig(TypedDict): ...


class MetaLearnerConfig(TypedDict): ...


class WeaselConfig(TypedDict):
    first_order: bool
    update_param_rate: float
    tune_epochs: int
    tune_val_freq: int | None
    tune_multi_step: bool


class ProtoSegConfig(TypedDict):
    multi_pred: bool
    embedding_size: int


class GuidedNetsConfig(TypedDict):
    embedding_size: int


class ConfigBase(TypedDict):
    data: DataConfig
    learn: LearnConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    log: LogConfig
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
