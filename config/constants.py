from typing import TypedDict


class FilenamesDict(TypedDict):
    log_folder: str
    model_onnx: str
    configuration: str
    configuration_diff: str
    profile: str
    dummy_file: str


FILENAMES: FilenamesDict = {
    "log_folder": "logs",
    "model_onnx": "model.onnx",
    "configuration": "configuration.json",
    "configuration_diff": "configuration_diff.json",
    "profile": "profile",
    "dummy_file": "dummy_sign",
}

WANDB_DIR = "wandb"
WANDB_ENTITY = "pandegaaz"