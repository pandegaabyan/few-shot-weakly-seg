from config.config_type import AllConfig, DataConfig, FWSConfig, TrainConfig, SaveConfig, WeaselConfig

data_config: DataConfig = {
    'num_classes': 2,
    'num_channels': 3,
    'num_workers': 0,
    'batch_size': 8,
    'resize_to': (256, 256)
}

fws_config: FWSConfig = {
    'list_shots': [1, 5, 10, 20],
    'list_sparsity_point': [1, 5, 10, 20],
    'list_sparsity_grid': [8, 12, 16, 20],
    'list_sparsity_contour': [0.05, 0.10, 0.25, 0.50, 1.00],
    'list_sparsity_skeleton': [0.05, 0.10, 0.25, 0.50, 1.00],
    'list_sparsity_region': [0.05, 0.10, 0.25, 0.50, 1.00]
}

train_config: TrainConfig = {
    'use_gpu': True,
    'epoch_num': 200,
    'lr': 1e-3,
    'lr_scheduler_step_size': 150,
    'lr_scheduler_gamma': 0.2,
    'weight_decay': 5e-5,
    'momentum': 0.9,
    'snapshot': '',
    'test_freq': 200,
    'n_metatasks_iter': 2
}

save_config: SaveConfig = {
    'ckpt_path': './ckpt/',
    'output_path': './outputs/',
    'exp_name': ''
}

weasel_config: WeaselConfig = {
    'first_order': False,
    'step_size': 0.3,
    'tuning_epochs': 40,
    'tuning_freq': 4
}

all_config: AllConfig = {
    'data': data_config,
    'fws': fws_config,
    'train': train_config,
    'save': save_config,
    'weasel': weasel_config
}
