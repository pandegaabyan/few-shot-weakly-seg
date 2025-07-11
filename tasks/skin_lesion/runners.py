from typing import Type

import optuna

from config.config_maker import gen_id
from config.config_type import (
    ConfigSimpleLearner,
    ConfigUnion,
)
from config.optuna import OptunaConfig
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.losses import CustomLoss
from learners.metrics import BinaryIoUMetric
from learners.simple_learner import SimpleLearner
from learners.simple_unet import SimpleUnet
from learners.typings import (
    DatasetLists,
    SimpleLearnerKwargs,
)
from runners.runner import Runner
from tasks.skin_lesion.datasets import (
    ISIC16SimpleDataset,
    ISIC17SimpleDataset,
    ISIC18SimpleDataset,
    PH2SimpleDataset,
)


def suggest_basic(config: ConfigUnion, trial: optuna.Trial) -> dict:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    beta1_comp = trial.suggest_float("beta1_comp", 1e-2, 1, log=True)
    beta2_comp = trial.suggest_float("beta2_comp", 1e-4, 1e-2, log=True)
    betas = (1 - beta1_comp, 1 - beta2_comp)
    lowest_gamma = (1e-10 / lr) ** (
        config["scheduler"].get("step_size", 1) / config["learn"]["num_epochs"]
    )
    gamma = trial.suggest_float("gamma", lowest_gamma, 1, log=True)

    config["optimizer"]["lr"] = lr
    config["optimizer"]["weight_decay"] = weight_decay
    config["optimizer"]["betas"] = betas
    config["scheduler"]["gamma"] = gamma

    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "beta1": betas[0],
        "beta2": betas[1],
        "gamma": gamma,
    }


def parse_basic(
    config: ConfigUnion, hyperparams: dict[str, bool | int | float | str]
) -> dict:
    lr = hyperparams.get("lr")
    weight_decay = hyperparams.get("weight_decay")
    beta1_comp = hyperparams.get("beta1_comp")
    beta2_comp = hyperparams.get("beta2_comp")
    gamma = hyperparams.get("gamma")

    important_config = {}
    if isinstance(lr, float):
        config["optimizer"]["lr"] = lr
        important_config["lr"] = lr
    if isinstance(weight_decay, float):
        config["optimizer"]["weight_decay"] = weight_decay
        important_config["weight_decay"] = weight_decay
    if isinstance(beta1_comp, float) and isinstance(beta2_comp, float):
        betas = (1 - beta1_comp, 1 - beta2_comp)
        config["optimizer"]["betas"] = betas
        important_config["beta1"] = betas[0]
        important_config["beta2"] = betas[1]
    if isinstance(gamma, float):
        config["scheduler"]["gamma"] = gamma
        important_config["gamma"] = gamma

    return important_config


class SimpleRunner(Runner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[SimpleLearner], SimpleLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)

        kwargs: SimpleLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": (CustomLoss, {"mode": "bce"}),
            "metric": (BinaryIoUMetric, {}),
            "optuna_trial": optuna_trial,
        }

        return SimpleUnet, kwargs

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        config: ConfigSimpleLearner = self.config  # type: ignore

        if optuna_trial is not None:
            important_config = suggest_basic(config, optuna_trial)
        else:
            important_config = parse_basic(
                config, self.optuna_config.get("hyperparams", {})
            )

        variable_max_batch = 32
        variable_epochs = 50
        homogen_batch = 10
        homogen_thresholds = (0.7, 0.8)
        homogen_count = 30
        homogen_epochs = 100
        if self.mode in ["profile-fit", "profile-test"]:
            config["learn"]["val_freq"] = 1
            profile_id = config["learn"].get("profile_id", None)
            assert profile_id is not None
            important_config["profile"] = profile_id
        if self.mode == "profile-fit":
            if self.number_of_multi < variable_max_batch:
                config["learn"]["num_epochs"] = variable_epochs
                batch_size = self.number_of_multi + 1
                config["data"]["batch_size"] = batch_size
                important_config["batch_size"] = batch_size
            else:
                config["data"]["batch_size"] = homogen_batch
                config["learn"]["num_epochs"] = homogen_epochs
                if self.number_of_multi < (variable_max_batch + homogen_count):
                    stop_threshold = homogen_thresholds[0]
                else:
                    stop_threshold = homogen_thresholds[1]
                config["callbacks"]["stop_threshold"] = stop_threshold
                important_config["stop_threshold"] = stop_threshold
            if self.number_of_multi == (variable_max_batch + 2 * homogen_count - 1):
                self.last_of_multi = True
        if self.mode == "profile-test":
            batch_size = self.number_of_multi + 1
            config["data"]["batch_size"] = batch_size
            important_config["batch_size"] = batch_size
            if self.number_of_multi == (variable_max_batch - 1):
                self.last_of_multi = True

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = self.learner_type + " " + gen_id(5)
        config["sampler_params"] = {
            "n_startup_trials": 20,
            "n_ei_candidates": 30,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "seed": self.seed,
        }
        config["pruner_params"] = {
            "min_resource": 10,
            "max_resource": self.config["learn"]["num_epochs"],
            "reduction_factor": 2,
            "bootstrap_count": 2,
        }
        config["pruner_patience"] = 5
        if not self.dummy:
            config["num_folds"] = 3
            config["timeout_sec"] = 8 * 3600
        return config

    def make_dataset_lists(
        self, val_fold: int, dummy: bool
    ) -> DatasetLists[SimpleDataset, SimpleDatasetKwargs]:
        base_kwargs: SimpleDatasetKwargs = {
            "seed": self.seed,
            "split_val_fold": val_fold,
            "split_test_fold": 0,
            "cache_data": True,
        }
        if dummy:
            base_kwargs["max_items"] = 6

        isic16_kwargs: SimpleDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "ISIC16",
        }
        isic17_kwargs: SimpleDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "ISIC17",
        }
        isic18_kwargs: SimpleDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "ISIC18",
        }
        ph2_kwargs: SimpleDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "PH2",
        }

        test_mult = 2 if self.mode == "profile-test" else 1
        isic16 = (ISIC16SimpleDataset, isic16_kwargs)
        isic17 = (ISIC17SimpleDataset, isic17_kwargs)
        isic18 = (ISIC18SimpleDataset, isic18_kwargs)
        ph2 = (PH2SimpleDataset, ph2_kwargs)

        if self.dataset.startswith("cross"):
            isic17_kwargs["split_test_size"] = 1.0
            isic18_kwargs["split_test_size"] = 1.0
            if self.dataset.startswith("cross-2"):
                isic16_kwargs["split_val_size"] = 0.2
                val_dataset_list = [isic16]
            else:
                ph2_kwargs["split_val_size"] = 1.0
                val_dataset_list = [ph2]
            if self.dataset.endswith("17"):
                test_dataset_list = [isic17]
            elif self.dataset.endswith("18"):
                test_dataset_list = [isic18]
            else:
                test_dataset_list = [isic17, isic18]
            return {
                "dataset_list": [isic16],
                "val_dataset_list": val_dataset_list,
                "test_dataset_list": test_dataset_list * test_mult,
            }

        for kwargs in [isic16_kwargs, isic17_kwargs, isic18_kwargs, ph2_kwargs]:
            kwargs["split_val_size"] = 0.2
            kwargs["split_test_size"] = 0.2
        if self.dataset == "ISIC16":
            dataset_list = [isic16]
        elif self.dataset == "ISIC17":
            dataset_list = [isic17]
        elif self.dataset == "ISIC18":
            dataset_list = [isic18]
        elif self.dataset == "PH2":
            dataset_list = [ph2]
        elif self.dataset == "all":
            dataset_list = [isic16, isic17, isic18, ph2]
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        return {
            "dataset_list": dataset_list,
            "test_dataset_list": dataset_list * test_mult,
        }


def get_runner_class(learner: str) -> Type[Runner]:
    runner_name = learner.split("-")[0]
    if runner_name == "SL":
        return SimpleRunner
    else:
        raise ValueError(f"Unknown runner: {runner_name}")
