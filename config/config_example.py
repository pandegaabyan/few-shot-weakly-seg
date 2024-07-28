from config.config_type import (
    AllConfig,
    DataConfig,
    DataTuneConfig,
    LearnConfig,
    SaveConfig,
    WeaselConfig,
)

data_config: DataConfig = {
    "num_classes": 2,
    "num_channels": 3,
    "num_workers": 0,
    "batch_size": 8,
    "resize_to": (256, 256),
}

data_tune_config: DataTuneConfig = {
    "list_shots": [1, 5, 10, 20],
    "list_sparsity_point": [1, 5, 10, 20],
    "list_sparsity_grid": [8, 12, 16, 20],
    "list_sparsity_contour": [0.05, 0.10, 0.25, 0.50, 1.00],
    "list_sparsity_skeleton": [0.05, 0.10, 0.25, 0.50, 1.00],
    "list_sparsity_region": [0.05, 0.10, 0.25, 0.50, 1.00],
}

train_config: LearnConfig = {
    "should_resume": False,
    "use_gpu": True,
    "num_epochs": 200,
    "optimizer_lr": 1e-3,
    "optimizer_weight_decay": 5e-5,
    "optimizer_momentum": 0.9,
    "scheduler_step_size": 150,
    "scheduler_gamma": 0.2,
    "tune_freq": 200,
    "meta_used_datasets": 2,
    "meta_iterations": 5,
}

save_config: SaveConfig = {
    "ckpt_path": "./ckpt/",
    "output_path": "./outputs/",
    "exp_name": "",
}

weasel_config: WeaselConfig = {
    "use_first_order": False,
    "update_param_step_size": 0.3,
    "tune_epochs": 40,
    "tune_test_freq": 4,
}

all_config: AllConfig = {
    "data": data_config,
    "data_tune": data_tune_config,
    "learn": train_config,
    "save": save_config,
    "weasel": weasel_config,
}
