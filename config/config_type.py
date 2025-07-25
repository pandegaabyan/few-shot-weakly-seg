from typing import Literal, Union, get_args

from typing_extensions import NotRequired, Required, TypedDict

RunMode = Literal["fit-test", "fit", "test", "study", "profile-fit", "profile-test"]
LearnerType = Literal[
    "SL",
    "WS",
    "WS-ms",
    "WS-ori",
    "WS-fo",
    "WS-ms-fo",
    "WS-ori-fo",
    "PS",
    "PS-mp",
    "PS-ori",
    "PA",
]
ProfilerType = Literal[
    "simple", "advanced", "pytorch", "custom", "custom-1", "custom-10", None
]

run_modes: list[RunMode] = list(get_args(RunMode))
learner_types: list[LearnerType] = list(get_args(LearnerType))
profiler_types: list[ProfilerType] = list(get_args(ProfilerType))


class DataConfig(TypedDict):
    num_classes: int
    num_channels: int
    num_workers: int
    batch_size: int
    resize_to: tuple[int, int]


class LearnConfig(TypedDict):
    exp_name: str
    run_name: str
    num_epochs: int
    dummy: NotRequired[bool]
    seed: NotRequired[int]
    val_freq: NotRequired[int]
    cudnn_deterministic: NotRequired[bool | Literal["warn"]]
    cudnn_benchmark: NotRequired[bool]
    profiler: NotRequired[ProfilerType]
    profile_id: NotRequired[str | None]
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
    clean_on_end: bool


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
    log_metrics: bool
    log_system_metrics: bool
    watch_model: bool
    save_model: bool
    push_table_freq: int | None
    save_mask_only: bool
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


class PANetConfig(TypedDict):
    embedding_size: int
    par_weight: float


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


class ConfigPANet(ConfigMetaLearner):
    panet: PANetConfig


class ConfigGuidedNets(ConfigMetaLearner):
    guidednets: GuidedNetsConfig


ConfigUnion = Union[
    ConfigBase,
    ConfigSimpleLearner,
    ConfigMetaLearner,
    ConfigWeasel,
    ConfigProtoSeg,
    ConfigPANet,
    ConfigGuidedNets,
]
