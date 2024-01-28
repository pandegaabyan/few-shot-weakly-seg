from typing import TypedDict


class FilenamesDict(TypedDict):
    checkpoint_folder: str
    output_folder: str
    optimizer_state: str
    net_state: str
    checkpoint: str
    config: str
    dataset_config: str
    optimization_data: str
    net_text: str
    learn_log: str
    train_loss: str
    tuned_score: str
    train_val_loss_score: str
    test_score: str
    prediction_folder: str
    tune_masks_folder: str


class DefaultConfigsDict(TypedDict):
    optimizer_lr: float
    optimizer_betas: tuple[float, float]
    scheduler_step_size: int
    scheduler_gamma: float


FILENAMES: FilenamesDict = {
    "checkpoint_folder": "./ckpt/",
    "output_folder": "./outputs/",
    "optimizer_state": "meta_optimizer.pth",
    "net_state": "net.pth",
    "checkpoint": "checkpoint.json",
    "config": "config.json",
    "dataset_config": "dataset.json",
    "optimization_data": "optimization.json",
    "net_text": "net.txt",
    "learn_log": "learn.log",
    "train_loss": "train_loss.csv",
    "tuned_score": "tuned_score.csv",
    "train_val_loss_score": "train_val_loss_score.csv",
    "test_score": "test_score.csv",
    "prediction_folder": "predictions",
    "tune_masks_folder": "all_tune_masks",
}


DEFAULT_CONFIGS: DefaultConfigsDict = {
    "optimizer_lr": 0.001,
    "optimizer_betas": (0.9, 0.999),
    "scheduler_step_size": 30,
    "scheduler_gamma": 0.1,
}
