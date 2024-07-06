from typing import Iterator

from torch import nn, optim

from config.config_type import OptimizerConfig, SchedulerConfig


def make_optimizer_adam(
    config: OptimizerConfig,
    named_params: Iterator[tuple[str, nn.Parameter]],
    separate_bias: bool = False,
) -> optim.Optimizer:
    default_lr = 0.001
    default_betas = (0.9, 0.999)

    lr = config.get("lr", default_lr)
    weight_decay = config.get("weight_decay", 0)
    betas = config.get("betas", default_betas)

    if separate_bias:
        return optim.Adam(
            [
                {
                    "params": [
                        param for name, param in named_params if name[-4:] == "bias"
                    ],
                    "lr": config.get("lr_bias", lr),
                    "weight_decay": config.get("weight_decay_bias", weight_decay),
                },
                {
                    "params": [
                        param for name, param in named_params if name[-4:] != "bias"
                    ],
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            ],
            betas=betas,
        )

    return optim.Adam(
        [param for _, param in named_params],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )


def make_scheduler_step(
    optimizer: optim.Optimizer, config: SchedulerConfig
) -> optim.lr_scheduler.StepLR:
    default_step_size = 30
    default_gamma = 0.1

    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("step_size", default_step_size),
        gamma=config.get("gamma", default_gamma),
    )
