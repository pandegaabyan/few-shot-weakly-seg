from typing import TypedDict


class FilenamesDict(TypedDict):
    log_folder: str
    model_onnx: str
    configuration: str
    configuration_diff: str
    dummy_file: str


FILENAMES: FilenamesDict = {
    "log_folder": "./logs/",
    "model_onnx": "model.onnx",
    "configuration": "configuration.json",
    "configuration_diff": "configuration_diff.json",
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
