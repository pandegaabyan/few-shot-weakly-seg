from typing import Type

import optuna

from config.config_type import ConfigSimpleLearner, ConfigUnion
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


def suggest_basic(config: ConfigUnion, trial: optuna.Trial) -> dict:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    lr_bias_mult = trial.suggest_categorical("lr_bias_mult", [0.5, 1, 2])
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    beta1_comp = trial.suggest_float("beta1_comp", 1e-2, 1, log=True)
    beta2_comp = trial.suggest_float("beta2_comp", 1e-4, 1e-2, log=True)
    lowest_gamma = (1e-10 / lr) ** (
        config["scheduler"].get("step_size", 1) / config["learn"]["num_epochs"]
    )
    gamma = trial.suggest_float("gamma", lowest_gamma, 1, log=True)

    config["optimizer"]["lr"] = lr
    config["optimizer"]["lr_bias_mult"] = lr_bias_mult
    config["optimizer"]["weight_decay"] = weight_decay
    config["optimizer"]["betas"] = (1 - beta1_comp, 1 - beta2_comp)
    config["scheduler"]["gamma"] = gamma

    return {
        "lr": config["optimizer"].get("lr"),
        "weight_decay": config["optimizer"].get("weight_decay"),
        "beta1": config["optimizer"].get("betas", (None, None))[0],
        "beta2": config["optimizer"].get("betas", (None, None))[1],
        "gamma": config["scheduler"].get("gamma"),
    }


class SimpleRunner(Runner):
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
            important_config = suggest_basic(cfg, optuna_trial)

        kwargs: SimpleLearnerKwargs = {
            "config": cfg,
            "dataset_list": dataset_list,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config.update(
            {
                "dataset": self.get_names_from_dataset_list(dataset_list),
            }
        )

        return SimpleUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Simple RIM-ONE"
        config["sampler_params"] = {
            "n_startup_trials": 20,
            "n_ei_candidates": 30,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "seed": 0,
        }
        config["pruner_params"] = {
            "reduction_factor": 2,
            "bootstrap_count": 2,
        }
        config["pruner_patience"] = 10
        if not self.dummy:
            config["num_folds"] = 3
            config["timeout_sec"] = 8 * 3600
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
