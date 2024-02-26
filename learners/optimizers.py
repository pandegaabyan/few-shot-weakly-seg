from typing import Iterator

from torch import nn, optim

from config.config_type import OptimizerConfig, SchedulerConfig
from config.constants import DEFAULT_CONFIGS


def make_optimizer_adam(
    config: OptimizerConfig, named_params: Iterator[tuple[str, nn.Parameter]]
) -> optim.Optimizer:
    adam_optimizer = optim.Adam(
        [
            {
                "params": [
                    param for name, param in named_params if name[-4:] == "bias"
                ],
                "lr": config.get("lr_bias"),
                "weight_decay": config.get("weight_decay_bias"),
            },
            {
                "params": [
                    param for name, param in named_params if name[-4:] != "bias"
                ],
                "lr": config.get("lr"),
                "weight_decay": config.get("weight_decay"),
            },
        ],
        betas=config.get("betas", DEFAULT_CONFIGS["optimizer_betas"]),
    )
    return adam_optimizer


def make_scheduler_step(
    optimizer: optim.Optimizer, config: SchedulerConfig
) -> optim.lr_scheduler.StepLR:
    step_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("step_size", DEFAULT_CONFIGS["scheduler_step_size"]),
        gamma=config.get("gamma", DEFAULT_CONFIGS["scheduler_gamma"]),
    )
    return step_scheduler
