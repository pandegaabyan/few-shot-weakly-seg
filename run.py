from typing import Type

import click
import optuna

from config.config_maker import make_config
from config.config_type import ConfigSimpleLearner, ConfigUnion, RunMode
from config.optuna import OptunaConfig
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.simple_unet import SimpleUnet
from learners.typings import BaseLearnerKwargs, SimpleLearnerKwargs
from runners.runner import Runner
from tasks.optic_disc_cup.datasets import DrishtiSimpleDataset, RimOneSimpleDataset
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU
from utils.logging import (
    check_git_clean,
)
from utils.utils import parse_string


class MyRunner(Runner):
    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[BaseLearner], BaseLearnerKwargs, dict]:
        dataset_list = [self.make_rim_one_dataset(dataset_fold)]
        for ds in dataset_list:
            ds[1]["max_items"] = 6 if dummy else None

        cfg: ConfigSimpleLearner = config  # type: ignore
        if optuna_trial is not None:
            lr = optuna_trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            lr_bias_mult = optuna_trial.suggest_categorical("lr_bias_mult", [0.5, 1, 2])
            weight_decay = optuna_trial.suggest_float(
                "weight_decay", 1e-10, 1e-3, log=True
            )
            beta1_comp = optuna_trial.suggest_float("beta1_comp", 1e-2, 1, log=True)
            beta2_comp = optuna_trial.suggest_float("beta2_comp", 1e-4, 1e-2, log=True)
            lowest_gamma = (1e-10 / lr) ** (
                cfg["scheduler"].get("step_size", 1) / cfg["learn"]["num_epochs"]
            )
            gamma = optuna_trial.suggest_float("gamma", lowest_gamma, 1, log=True)
            cfg["optimizer"]["lr"] = lr
            cfg["optimizer"]["lr_bias_mult"] = lr_bias_mult
            cfg["optimizer"]["weight_decay"] = weight_decay
            cfg["optimizer"]["betas"] = (1 - beta1_comp, 1 - beta2_comp)
            cfg["scheduler"]["gamma"] = gamma

        kwargs: SimpleLearnerKwargs = {
            "config": cfg,
            "dataset_list": dataset_list,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config = {
            "dataset": self.get_names_from_dataset_list(dataset_list),
            "lr": cfg["optimizer"].get("lr"),
            "weight_decay": cfg["optimizer"].get("weight_decay"),
            "beta1": cfg["optimizer"].get("betas", (None, None))[0],
            "beta2": cfg["optimizer"].get("betas", (None, None))[1],
            "gamma": cfg["scheduler"].get("gamma"),
        }

        return SimpleUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Simple RIM-ONE Median"
        config["sampler_params"] = {
            "n_startup_trials": 20,
            "n_ei_candidates": 30,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "seed": 0,
        }
        config["pruner"] = "median"
        config["pruner_params"] = {
            "n_startup_trials": 20,
            "n_warmup_steps": 10,
            "n_min_trials": 2,
        }
        config["pruner_patience"] = 5
        if not self.dummy:
            config["num_folds"] = 3
            config["timeout_sec"] = 6 * 3600
        return config

    def make_rim_one_dataset(
        self,
        val_fold: int = 0,
    ) -> tuple[Type[SimpleDataset], SimpleDatasetKwargs]:
        rim_one_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "max_items": None,
            "split_val_size": 0.2,
            "split_val_fold": val_fold,
            "split_test_size": 0.2,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "RIM-ONE",
        }

        return (RimOneSimpleDataset, rim_one_kwargs)

    def make_drishti_dataset(
        self,
        val_fold: int = 0,
    ) -> tuple[Type[SimpleDataset], SimpleDatasetKwargs]:
        drishti_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "max_items": None,
            "split_val_size": 0.15,
            "split_val_fold": val_fold,
            "split_test_size": 0.15,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "DRISHTI",
        }

        return (DrishtiSimpleDataset, drishti_kwargs)


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fit-test", "fit", "test", "study"]),
    default="fit-test",
)
@click.option(
    "--configs",
    "-c",
    nargs=2,
    multiple=True,
    type=(str, str),
    default=[],
    help="(key, value) for overriding config, use '/' for nesting keys",
)
@click.option(
    "--optuna_configs",
    "-oc",
    nargs=2,
    multiple=True,
    type=(str, str),
    default=[],
    help="(key, value) for overriding optuna config",
)
def main(
    mode: RunMode,
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    configs: list[tuple[str, str]],
    optuna_configs: list[tuple[str, str]],
):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")

    config = make_config(
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner="simple"
    )
    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    my_runner = MyRunner(config, dummy, resume)

    if mode in ["fit-test", "fit", "test"]:
        my_runner.run_fit_test(mode == "fit", mode == "test")

    for key, value in optuna_configs:
        my_runner.optuna_config[key] = parse_string(value)

    if mode in ["study"]:
        my_runner.run_study()


if __name__ == "__main__":
    main()
