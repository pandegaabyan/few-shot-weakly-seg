from typing import Iterator, Union

from pytorch_lightning.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from torch import nn, optim

from config.config_type import OptimizerConfig, SchedulerConfig
from learners.typings import Optimizer
from utils.logging import get_name_from_instance


def make_optimizer_adam(
    config: OptimizerConfig,
    named_params: Iterator[tuple[str, nn.Parameter]],
) -> Optimizer:
    default_lr = 0.001
    default_betas = (0.9, 0.999)
    lr = config.get("lr", default_lr)

    return optim.Adam(
        [
            {
                "params": [
                    param for name, param in named_params if name[-4:] == "bias"
                ],
                "lr": config.get("lr_bias_mult", 1) * lr,
            },
            {
                "params": [
                    param for name, param in named_params if name[-4:] != "bias"
                ],
                "lr": lr,
                "weight_decay": config.get("weight_decay", 0),
            },
        ],
        betas=config.get("betas", default_betas),
    )


def make_scheduler_step(
    optimizer: Optimizer, config: SchedulerConfig
) -> optim.lr_scheduler.StepLR:
    default_step_size = 30
    default_gamma = 0.1

    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("step_size", default_step_size),
        gamma=config.get("gamma", default_gamma),
    )


def get_optimizer_and_scheduler_names(
    opt_sched: OptimizerLRScheduler,
) -> tuple[list[str], list[str]]:
    def get_sched_from_union(
        union: Union[LRSchedulerTypeUnion, LRSchedulerConfigType]
        | Union[LRSchedulerTypeUnion, LRSchedulerConfig],
    ) -> str:
        if isinstance(union, dict):
            return "LRSchedulerConfigType"
        return get_name_from_instance(union)

    def get_opt_and_sched_from_config(
        config: OptimizerLRSchedulerConfig,
    ) -> tuple[str, str]:
        sched = config.get("lr_scheduler")
        sched = get_sched_from_union(sched) if sched else ""
        return get_name_from_instance(config["optimizer"]), sched

    if isinstance(opt_sched, optim.Optimizer):
        return [get_name_from_instance(opt_sched)], [""]
    if isinstance(opt_sched, dict):
        opt, sched = get_opt_and_sched_from_config(opt_sched)
        return [opt], [sched]
    if (
        isinstance(opt_sched, tuple)
        and isinstance(opt_sched[0], (list, tuple))
        and isinstance(opt_sched[1], (list, tuple))
    ):
        opts, scheds = [], []
        for opt in opt_sched[0]:
            opts.append(get_name_from_instance(opt))
        for sched in opt_sched[1]:
            scheds.append(get_sched_from_union(sched))
        return opts, scheds
    if isinstance(opt_sched, list) or isinstance(opt_sched, tuple):
        opts, scheds = [], []
        for item in opt_sched:
            if isinstance(item, optim.Optimizer):
                opts.append(get_name_from_instance(item))
                scheds.append("")
            elif isinstance(item, dict):
                opt, sched = get_opt_and_sched_from_config(item)
                opts.append(opt)
                scheds.append(sched)
        return opts, scheds
    return [], []
