from config.config_type import (
    AllConfig,
    DataConfig,
    DataTuneConfig,
    LearnConfig,
    WeaselConfig,
    ProtoSegConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
)

data_config: DataConfig = {
    "num_classes": 2,
    "num_channels": 3,
    "num_workers": 0,
    "batch_size": 8,
    "resize_to": (256, 256),
}

data_tune_config: DataTuneConfig = {
    "shot_list": [1, 5, 10, 20],
    "sparsity_dict": {
        "point": [1, 5, 10, 20],
        "grid": [8, 12, 16, 20],
        "contour": [0.05, 0.10, 0.25, 0.50, 1.00],
        "skeleton": [0.05, 0.10, 0.25, 0.50, 1.00],
        "region": [0.05, 0.10, 0.25, 0.50, 1.00],
    },
}

train_config: LearnConfig = {
    "should_resume": False,
    "use_gpu": True,
    "num_epochs": 200,
    "tune_freq": 200,
    "exp_name": "",
}

loss_config: LossConfig = {"type": "ce", "ignored_index": -1}

optimizer_config: OptimizerConfig = {
    "lr": 1e-3,
    "lr_bias": 2 * 1e-3,
    "weight_decay": 5e-5,
    "weight_decay_bias": 0,
    "betas": (0.9, 0.99),
}

scheduler_config: SchedulerConfig = {"step_size": 150, "gamma": 0.2}

weasel_config: WeaselConfig = {
    "use_first_order": False,
    "update_param_step_size": 0.3,
    "tune_epochs": 40,
    "tune_test_freq": 4,
}

protoseg_config: ProtoSegConfig = {"embedding_size": 4}

all_config: AllConfig = {
    "data": data_config,
    "data_tune": data_tune_config,
    "learn": train_config,
    "loss": loss_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
    "weasel": weasel_config,
    "protoseg": protoseg_config,
}
