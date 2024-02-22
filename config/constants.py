from typing import TypedDict


class FilenamesDict(TypedDict):
    checkpoint_folder: str
    log_folder: str
    tensorboard_folder: str
    model_onnx: str
    configuration: str
    configuration_diff: str
    sweep_config: str
    recent_runs: str
    dummy_file: str


FILENAMES: FilenamesDict = {
    "checkpoint_folder": "./ckpt/",
    "log_folder": "./logs/",
    "tensorboard_folder": "./tensorboard/",
    "model_onnx": "model.onnx",
    "configuration": "configuration.json",
    "configuration_diff": "configuration_diff.json",
    "sweep_config": "sweep_config.json",
    "recent_runs": "recent_runs.txt",
    "dummy_file": "dummy_sign",
}


class WandbSettingsDict(TypedDict):
    entity: str
    project: str
    dummy_project: str
    dir: str


WANDB_SETTINGS: WandbSettingsDict = {
    "entity": "pandegaaz",
    "project": "few-shot-weakly-seg",
    "dummy_project": "few-shot-weakly-seg-dummy",
    "dir": "wandb",
}


class DefaultConfigsDict(TypedDict):
    optimizer_lr: float
    optimizer_betas: tuple[float, float]
    scheduler_step_size: int
    scheduler_gamma: float


DEFAULT_CONFIGS: DefaultConfigsDict = {
    "optimizer_lr": 0.001,
    "optimizer_betas": (0.9, 0.999),
    "scheduler_step_size": 30,
    "scheduler_gamma": 0.1,
}
