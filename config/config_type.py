from typing import Literal, Union

from typing_extensions import NotRequired, TypedDict


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
    tensorboard_graph: NotRequired[bool]
    ref_ckpt_path: NotRequired[str | None]


class LossConfig(TypedDict, total=False):
    type: str
    ignored_index: int


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
    progress_leave: bool
    ckpt_monitor: str
    ckpt_mode: Literal["min", "max"]
    ckpt_top_k: int


class WandbConfig(TypedDict):
    run_id: str
    tags: list[str]
    job_type: str | None
    log_model: bool
    watch_model: bool
    push_table_freq: int | None
    sweep_metric: NotRequired[tuple[str, Literal["maximize", "minimize"]]] | None
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
    loss: LossConfig
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
