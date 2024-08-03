from typing import Type

import optuna
from typing_extensions import Generic, Required, TypedDict

from config.config_type import (
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigUnion,
    ConfigWeasel,
)
from config.optuna import OptunaConfig
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import FewSparseDatasetKwargs, SimpleDatasetKwargs
from learners.protoseg_learner import ProtosegLearner
from learners.protoseg_unet import ProtosegUnet
from learners.simple_learner import SimpleLearner
from learners.simple_unet import SimpleUnet
from learners.typings import (
    DatasetClass,
    DatasetKwargs,
    ProtoSegLearnerKwargs,
    SimpleLearnerKwargs,
    WeaselLearnerKwargs,
)
from learners.weasel_learner import WeaselLearner
from learners.weasel_unet import WeaselUnet
from runners.runner import Runner
from tasks.optic_disc_cup.datasets import (
    DrishtiTestFSDataset,
    DrishtiTrainFSDataset,
    RefugeTestFSDataset,
    RefugeTestSimpleDataset,
    RefugeTrainFSDataset,
    RefugeTrainSimpleDataset,
    RefugeValFSDataset,
    RefugeValSimpleDataset,
    RimOne3TestFSDataset,
    RimOne3TrainFSDataset,
)
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU


class DatasetLists(TypedDict, Generic[DatasetClass, DatasetKwargs], total=False):
    dataset_list: Required[list[tuple[Type[DatasetClass], DatasetKwargs]]]
    val_dataset_list: list[tuple[Type[DatasetClass], DatasetKwargs]]
    test_dataset_list: list[tuple[Type[DatasetClass], DatasetKwargs]]


def get_names_from_dataset_list(dataset_lists: DatasetLists) -> dict[str, str]:
    return {
        key.replace("_list", ""): ",".join(
            [
                (kwargs.get("dataset_name") or "NO-NAME")
                for _, kwargs in dataset_lists[key]
            ]
        )
        for key in dataset_lists
    }


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
        dataset_lists = self.make_dataset_lists(dataset_fold, dummy)

        cfg: ConfigSimpleLearner = config  # type: ignore
        if optuna_trial is not None:
            important_config = suggest_basic(cfg, optuna_trial)
        else:
            important_config = {}

        kwargs: SimpleLearnerKwargs = {
            **dataset_lists,
            "config": cfg,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config.update(get_names_from_dataset_list(dataset_lists))

        return SimpleUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Simple REFUGE train-val"
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
            config["timeout_sec"] = 4 * 3600
        return config

    def make_dataset_lists(
        self, val_fold: int, dummy: bool
    ) -> DatasetLists[SimpleDataset, SimpleDatasetKwargs]:
        base_kwargs: SimpleDatasetKwargs = {
            "max_items": 6 if dummy else None,
            "seed": 0,
            "split_val_fold": val_fold,
            "split_test_fold": 0,
            "cache_data": True,
        }

        rim_one_3_train_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "RIM-ONE-3-train",
            "split_val_size": 0.2,
        }
        rim_one_3_test_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "RIM-ONE-3-test",
            "split_test_size": 1,
        }
        drishti_train_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "DRISHTI-GS-train",
            "split_val_size": 0.1,
        }
        drishti_test_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "DRISHTI-GS-test",
            "split_test_size": 1,
        }
        refuge_train_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "REFUGE-train",
        }
        refuge_val_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "REFUGE-val",
            "split_val_size": 1,
        }
        refuge_test_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "REFUGE-test",
            "split_test_size": 1,
        }

        return {
            "dataset_list": [
                (RefugeTrainSimpleDataset, refuge_train_kwargs),
            ],
            "val_dataset_list": [
                (RefugeValSimpleDataset, refuge_val_kwargs),
            ],
            "test_dataset_list": [
                (RefugeTestSimpleDataset, refuge_test_kwargs),
            ],
        }


class MetaRunner(Runner):
    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["pruner"] = "median"
        config["num_folds"] = 2
        config["num_trials"] = 3
        return config

    def make_dataset_lists(
        self, query_fold: int, dummy: bool
    ) -> DatasetLists[FewSparseDataset, FewSparseDatasetKwargs]:
        base_kwargs: FewSparseDatasetKwargs = {
            "max_items": 6 if dummy else None,
            "seed": 0,
            "cache_data": True,
            "query_batch_size": self.config["data"]["batch_size"],
            "split_query_fold": query_fold,
        }

        rim_one_3_train_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "RIM-ONE-3-train",
            "split_val_size": 1,
        }
        rim_one_3_test_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "RIM-ONE-3-test",
            "split_test_size": 1,
        }
        drishti_train_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "DRISHTI-GS-train",
            "split_val_size": 1,
        }
        drishti_test_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "DRISHTI-GS-test",
            "split_test_size": 1,
        }
        refuge_train_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "REFUGE-train",
        }
        refuge_val_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "REFUGE-val",
            "split_val_size": 1,
        }
        refuge_test_kwargs: FewSparseDatasetKwargs = {
            **base_kwargs,
            "dataset_name": "REFUGE-test",
            "split_test_size": 1,
        }

        return {
            "dataset_list": [
                (RefugeTrainFSDataset, refuge_train_kwargs),
            ],
            "val_dataset_list": [
                (RefugeValFSDataset, refuge_val_kwargs),
                (RimOne3TrainFSDataset, rim_one_3_train_kwargs),
                (DrishtiTrainFSDataset, drishti_train_kwargs),
            ],
            "test_dataset_list": [
                (RefugeTestFSDataset, refuge_test_kwargs),
                (RimOne3TestFSDataset, rim_one_3_test_kwargs),
                (DrishtiTestFSDataset, drishti_test_kwargs),
            ],
        }


class WeaselRunner(MetaRunner):
    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[WeaselLearner], WeaselLearnerKwargs, dict]:
        dataset_lists = self.make_dataset_lists(dataset_fold, dummy)

        cfg: ConfigWeasel = config  # type: ignore
        if optuna_trial is not None:
            important_config = suggest_basic(cfg, optuna_trial)
        else:
            important_config = {}

        kwargs: WeaselLearnerKwargs = {
            **dataset_lists,
            "config": cfg,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config.update(get_names_from_dataset_list(dataset_lists))

        return WeaselUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Weasel"
        return config


class ProtosegRunner(MetaRunner):
    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[ProtosegLearner], ProtoSegLearnerKwargs, dict]:
        dataset_lists = self.make_dataset_lists(dataset_fold, dummy)

        cfg: ConfigProtoSeg = config  # type: ignore
        if optuna_trial is not None:
            important_config = suggest_basic(cfg, optuna_trial)
        else:
            important_config = {}

        kwargs: ProtoSegLearnerKwargs = {
            **dataset_lists,
            "config": cfg,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        important_config.update(get_names_from_dataset_list(dataset_lists))

        return ProtosegUnet, kwargs, important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["study_name"] = "Protoseg"
        return config
