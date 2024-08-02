from typing import Sequence, Type

import optuna

from config.config_type import (
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigUnion,
    ConfigWeasel,
)
from config.optuna import OptunaConfig
from data.base_dataset import BaseDataset
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import BaseDatasetKwargs, FewSparseDatasetKwargs, SimpleDatasetKwargs
from learners.protoseg_learner import ProtosegLearner
from learners.protoseg_unet import ProtosegUnet
from learners.simple_learner import SimpleLearner
from learners.simple_unet import SimpleUnet
from learners.typings import (
    ProtoSegLearnerKwargs,
    SimpleLearnerKwargs,
    WeaselLearnerKwargs,
)
from learners.weasel_learner import WeaselLearner
from learners.weasel_unet import WeaselUnet
from runners.runner import Runner
from tasks.optic_disc_cup.datasets import (
    DrishtiFSDataset,
    DrishtiSimpleDataset,
    OrigaSimpleDataset,
    PapilaSimpleDataset,
    RimOneFSDataset,
    RimOneSimpleDataset,
)
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU


def update_datasets_for_dummy(
    *dataset_lists: Sequence[tuple[Type[BaseDataset], BaseDatasetKwargs]],
):
    for ds_list in dataset_lists:
        for ds in ds_list:
            ds[1]["max_items"] = 6


def suggest_basic(config: ConfigUnion, trial: optuna.Trial) -> dict:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    beta1_comp = trial.suggest_float("beta1_comp", 1e-2, 1, log=True)
    beta2_comp = trial.suggest_float("beta2_comp", 1e-4, 1e-2, log=True)
    lowest_gamma = (1e-10 / lr) ** (
        config["scheduler"].get("step_size", 1) / config["learn"]["num_epochs"]
    )
    gamma = trial.suggest_float("gamma", lowest_gamma, 1, log=True)

    config["optimizer"]["lr"] = lr
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
    ) -> tuple[Type[SimpleLearner], SimpleLearnerKwargs, dict]:
        dataset_list = [self.make_papila_dataset(dataset_fold)]
        if dummy:
            update_datasets_for_dummy(dataset_list)

        cfg: ConfigSimpleLearner = config  # type: ignore
        if optuna_trial is not None:
            important_config = suggest_basic(cfg, optuna_trial)
        else:
            important_config = {}

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
        config["study_name"] = "Simple PAPILA"
        config["sampler_params"] = {
            "n_startup_trials": 20,
            "n_ei_candidates": 30,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "seed": 0,
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
            config["timeout_sec"] = 12 * 3600
        return config

    def make_rim_one_dataset(
        self,
        val_fold: int = 0,
    ) -> tuple[Type[SimpleDataset], SimpleDatasetKwargs]:
        rim_one_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "split_val_size": 0.2,
            "split_val_fold": val_fold,
            "split_test_size": 0.2,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "RIM-ONE DL",
        }

        return (RimOneSimpleDataset, rim_one_kwargs)

    def make_drishti_dataset(
        self,
        val_fold: int = 0,
    ) -> tuple[Type[SimpleDataset], SimpleDatasetKwargs]:
        drishti_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "split_val_size": 0.15,
            "split_val_fold": val_fold,
            "split_test_size": 0.15,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "DRISHTI",
        }

        return (DrishtiSimpleDataset, drishti_kwargs)

    def make_origa_dataset(
        self,
        val_fold: int = 0,
    ) -> tuple[Type[SimpleDataset], SimpleDatasetKwargs]:
        origa_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "split_val_size": 0.2,
            "split_val_fold": val_fold,
            "split_test_size": 0.2,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "ORIGA",
        }

        return (OrigaSimpleDataset, origa_kwargs)

    def make_papila_dataset(
        self,
        val_fold: int = 0,
    ) -> tuple[Type[SimpleDataset], SimpleDatasetKwargs]:
        papila_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "split_val_size": 0.2,
            "split_val_fold": val_fold,
            "split_test_size": 0.2,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "PAPILA",
        }

        return (PapilaSimpleDataset, papila_kwargs)


class MetaRunner(Runner):
    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["pruner"] = "median"
        config["num_folds"] = 2
        config["num_trials"] = 3
        return config

    def make_rim_one_dataset(
        self,
        query_fold: int = 0,
    ) -> tuple[Type[FewSparseDataset], FewSparseDatasetKwargs]:
        rim_one_kwargs: FewSparseDatasetKwargs = {
            "seed": 0,
            "cache_data": True,
            "dataset_name": "RIM-ONE DL",
            "shot_options": [2],
            "sparsity_options": [("random", "random")],
            "query_batch_size": 2,
            "split_query_size": 0.5,
            "split_query_fold": query_fold,
            "num_iterations": 2,
        }

        return (RimOneFSDataset, rim_one_kwargs)

    def make_drishti_dataset(
        self,
        query_fold: int = 0,
    ) -> tuple[Type[FewSparseDataset], FewSparseDatasetKwargs]:
        drishti_kwargs: FewSparseDatasetKwargs = {
            "seed": 0,
            "split_val_size": 0.6,
            "split_val_fold": 0,
            "split_test_size": 0.4,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "DRISHTI",
            "shot_options": [2],
            "sparsity_options": [("random", "random")],
            "query_batch_size": 2,
            "split_query_size": 0.5,
            "split_query_fold": query_fold,
            "num_iterations": 2,
        }

        return (DrishtiFSDataset, drishti_kwargs)


class WeaselRunner(MetaRunner):
    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[WeaselLearner], WeaselLearnerKwargs, dict]:
        dataset_list = [self.make_rim_one_dataset(dataset_fold)]
        val_dataset_list = [self.make_drishti_dataset(dataset_fold)]
        if dummy:
            update_datasets_for_dummy(dataset_list, val_dataset_list)

        cfg: ConfigWeasel = config  # type: ignore
        if optuna_trial is not None:
            important_config = suggest_basic(cfg, optuna_trial)
        else:
            important_config = {}

        kwargs: WeaselLearnerKwargs = {
            "config": cfg,
            "dataset_list": dataset_list,
            "val_dataset_list": val_dataset_list,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config.update(
            {
                "dataset": self.get_names_from_dataset_list(dataset_list),
                "val_dataset": self.get_names_from_dataset_list(val_dataset_list),
            }
        )

        return WeaselUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Weasel RIM-ONE to DRISHTI"
        return config


class ProtosegRunner(MetaRunner):
    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[ProtosegLearner], ProtoSegLearnerKwargs, dict]:
        dataset_list = [self.make_rim_one_dataset(dataset_fold)]
        val_dataset_list = [self.make_drishti_dataset(dataset_fold)]
        if dummy:
            update_datasets_for_dummy(dataset_list, val_dataset_list)

        cfg: ConfigProtoSeg = config  # type: ignore
        if optuna_trial is not None:
            important_config = suggest_basic(cfg, optuna_trial)
        else:
            important_config = {}

        kwargs: ProtoSegLearnerKwargs = {
            "config": cfg,
            "dataset_list": dataset_list,
            "val_dataset_list": val_dataset_list,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config.update(
            {
                "dataset": self.get_names_from_dataset_list(dataset_list),
                "val_dataset": self.get_names_from_dataset_list(val_dataset_list),
            }
        )

        return ProtosegUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Protoseg RIM-ONE to DRISHTI"
        return config
