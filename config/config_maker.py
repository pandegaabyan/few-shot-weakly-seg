from typing import Literal

import nanoid

from config.config_type import (
    AllConfig,
    DataConfig,
    DataTuneConfig,
    LearnConfig,
    ProtosegConfig,
    SaveConfig,
    WeaselConfig,
)

data_config: DataConfig = {
    "num_classes": 3,
    "num_channels": 3,
    "num_workers": 0,
    "batch_size": 1,
    "resize_to": (256, 256),
}

data_tune_config: DataTuneConfig = {
    "list_shots": [5],
    "list_sparsity_point": [50],
    "list_sparsity_grid": [30],
    "list_sparsity_contour": [1],
    "list_sparsity_skeleton": [1],
    "list_sparsity_region": [1],
}

# data_tune_config: DataTuneConfig = {
#     "list_shots": [1, 5, 10, 20],
#     "list_sparsity_point": [1, 5, 10, 20],
#     "list_sparsity_grid": [8, 12, 16, 20],
#     "list_sparsity_contour": [0.05, 0.10, 0.25, 0.50, 1.00],
#     "list_sparsity_skeleton": [0.05, 0.10, 0.25, 0.50, 1.00],
#     "list_sparsity_region": [0.05, 0.10, 0.25, 0.50, 1.00],
# }

learn_config: LearnConfig = {
    "should_resume": False,
    "use_gpu": True,
    "num_epochs": 6,
    "optimizer_lr": 1e-3,
    "optimizer_weight_decay": 5e-5,
    "optimizer_momentum": 0.9,
    # "scheduler_step_size": 150,
    "scheduler_step_size": 50,
    "scheduler_gamma": 0.2,
    "tune_freq": 3,
    "meta_used_datasets": 1,
    "meta_iterations": 2,
}

save_config: SaveConfig = {
    "ckpt_path": "./ckpt/",
    "output_path": "./outputs/",
    "exp_name": "",
    "minimal_save": False,
}

weasel_config: WeaselConfig = {
    "use_first_order": False,
    "update_param_rate": 0.3,
    "tune_epochs": 4,
    "tune_test_freq": 2,
}

protoseg_config: ProtosegConfig = {
    "embedding_size": 3,
}

all_config: AllConfig = {
    "data": data_config,
    "data_tune": data_tune_config,
    "learn": learn_config,
    "save": save_config,
    "weasel": weasel_config,
    "protoseg": protoseg_config,
}


def make_exp_name(learner: str | None = None, dummy: bool = False) -> str:
    name_prefix = (
        "WS" if learner == "weasel" else "PS" if learner == "protoseg" else "N"
    )
    exp_name = f"{name_prefix}{' D ' if dummy else ' '}{nanoid.generate(size=5)}"
    return exp_name


def make_config(
    learner: Literal["weasel", "protoseg", None] = None,
    mode: Literal["fit", "study"] = "fit",
    dummy: bool = False,
) -> AllConfig:
    if dummy:
        if learner == "weasel":
            all_config["data"]["batch_size"] = 3
        elif learner == "protoseg":
            all_config["data"]["batch_size"] = 5
    else:
        all_config["data"]["num_workers"] = 3
        all_config["learn"]["num_epochs"] = 200
        all_config["learn"]["tune_freq"] = 40
        all_config["learn"]["meta_iterations"] = 5
        all_config["weasel"]["tune_epochs"] = 40
        all_config["weasel"]["tune_test_freq"] = 8
        if learner == "weasel":
            all_config["data"]["batch_size"] = 16
        elif learner == "protoseg":
            all_config["data"]["batch_size"] = 32

    if mode == "study":
        all_config["save"]["minimal_save"] = True

    all_config["save"]["exp_name"] = make_exp_name(learner=learner, dummy=dummy)

    return all_config
