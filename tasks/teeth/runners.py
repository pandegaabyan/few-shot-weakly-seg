from typing import Any, Type

import optuna

from config.config_type import (
    ConfigMetaLearner,
    ConfigPANet,
    ConfigPASNet,
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigUnion,
    ConfigWeasel,
)
from config.optuna import OptunaConfig
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import FewSparseDatasetKwargs, SimpleDatasetKwargs
from learners.losses import CustomLoss
from learners.metrics import BaseMetric, DiceMetric
from learners.panet_learner import PANetLearner
from learners.pasnet_learner import PASNetLearner
from learners.protoseg_learner import ProtosegLearner
from learners.simple_learner import SimpleLearner
from learners.typings import (
    DatasetLists,
    PANetLearnerKwargs,
    PASNetLearnerKwargs,
    ProtoSegLearnerKwargs,
    SimpleLearnerKwargs,
    WeaselLearnerKwargs,
)
from learners.weasel_learner import WeaselLearner
from runners.runner import Runner
from tasks.teeth.datasets import (
    AdnanUmerFSDataset,
    AdnanUmerSimpleDataset,
    DualLabeledFSDataset,
    DualLabeledSimpleDataset,
    HITLFSDataset,
    HITLSimpleDataset,
    TuftsFSDataset,
    TuftsSimpleDataset,
    UFBA425FSDataset,
    UFBA425SimpleDataset,
    teeth_sparsity_params_256,
    teeth_sparsity_params_512,
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


def parse_basic(config: ConfigUnion, optuna_config: OptunaConfig) -> dict:
    hyperparams = optuna_config.get("hyperparams", {})
    lr = hyperparams.get("lr")
    weight_decay = hyperparams.get("weight_decay")
    beta1_comp = hyperparams.get("beta1_comp")
    beta2_comp = hyperparams.get("beta2_comp")
    gamma = hyperparams.get("gamma")

    if isinstance(lr, float):
        config["optimizer"]["lr"] = lr
    if isinstance(weight_decay, float):
        config["optimizer"]["weight_decay"] = weight_decay
    if isinstance(beta1_comp, float) and isinstance(beta2_comp, float):
        betas = (1 - beta1_comp, 1 - beta2_comp)
        config["optimizer"]["betas"] = betas
    if isinstance(gamma, float):
        config["scheduler"]["gamma"] = gamma

    important_config = {}
    if "lr" in config["optimizer"]:
        important_config["lr"] = config["optimizer"]["lr"]
    if "weight_decay" in config["optimizer"]:
        important_config["weight_decay"] = config["optimizer"]["weight_decay"]
    if "betas" in config["optimizer"]:
        important_config["beta1"] = config["optimizer"]["betas"][0]
        important_config["beta2"] = config["optimizer"]["betas"][1]
    if "gamma" in config["scheduler"]:
        important_config["gamma"] = config["scheduler"]["gamma"]

    return important_config


def suggest_or_parse_model(
    config: ConfigUnion, trial: optuna.Trial | None, optuna_config: OptunaConfig
) -> None:
    if trial is not None and (
        config["model"].get("arch") in ["deeplabv3", "deeplabv3plus"]
    ):
        config["model"]["backbone"] = trial.suggest_categorical(
            "backbone", ["mobilenetv2", "resnet50", "hrnetv2_32"]
        )
    else:
        backbone = optuna_config.get("hyperparams", {}).get("backbone")
        if isinstance(backbone, str):
            config["model"]["backbone"] = backbone
            if "arch" not in config["model"]:
                config["model"]["arch"] = "deeplabv3plus"


def define_loss_metric(
    config: ConfigUnion,
) -> tuple[
    tuple[Type[CustomLoss], dict[str, Any]], tuple[Type[BaseMetric], dict[str, Any]]
]:
    bg_weight = config["model"].get("bg_weight", 0.1)
    loss = (
        CustomLoss,
        {
            "mode": "ce",
            "ce_weights": [bg_weight] + [1.0] * (config["data"]["num_classes"] - 1),
        },
    )
    metric = (
        DiceMetric,
        {
            "num_classes": config["data"]["num_classes"],
            "average": "micro",
            "ignore_index": 0,
        },
    )
    return loss, metric


class SimpleRunner(Runner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[SimpleLearner], SimpleLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)
        loss, metric = define_loss_metric(self.config)

        kwargs: SimpleLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": loss,
            "metric": metric,
            "optuna_trial": optuna_trial,
        }

        return SimpleLearner, kwargs

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        config: ConfigSimpleLearner = self.config  # type: ignore

        suggest_or_parse_model(config, optuna_trial, self.optuna_config)

        if optuna_trial is not None:
            important_config = suggest_basic(config, optuna_trial)
        else:
            important_config = parse_basic(config, self.optuna_config)
        important_config = {**self.get_model_config(), **important_config}

        bg_weight = (
            optuna_trial.suggest_float("bg_weight", 0.01, 1.0, log=True)
            if optuna_trial is not None
            else self.optuna_config.get("hyperparams", {}).get("bg_weight", 0.1)
        )
        config["model"]["bg_weight"] = bg_weight  # type: ignore
        important_config["bg_weight"] = bg_weight

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["sampler_params"] = {
            "n_startup_trials": 10,
            "n_ei_candidates": 30,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "seed": self.seed,
        }
        config["pruner"] = "median"
        config["pruner_params"] = {
            "n_warmup_steps": 21,
            "interval_steps": 2,
            "n_min_trials": 3,
        }
        config["pruner_patience"] = 0
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
            "scaling": "simple",
            "cache_data": True,
        }
        if dummy:
            base_kwargs["size"] = 6

        dataset_names = [
            "HITL",
            "Adnan-Umer",
            "Dual-Labeled",
            "Tufts",
            "UFBA-425",
        ]
        dataset_classes = [
            HITLSimpleDataset,
            AdnanUmerSimpleDataset,
            DualLabeledSimpleDataset,
            TuftsSimpleDataset,
            UFBA425SimpleDataset,
        ]
        dataset_class_map = {
            name: clas for name, clas in zip(dataset_names, dataset_classes)
        }

        splitted_dataset = self.dataset.split(":")

        if splitted_dataset[0] in ["all", "all2"]:
            if splitted_dataset[0] == "all":
                train_names = ["HITL"]
                val_names = ["Tufts", "Dual-Labeled"]
                val_splits = {"Tufts": 0.1, "Dual-Labeled": 0.2}
            else:
                train_names = ["Dual-Labeled"]
                val_names = ["Tufts", "HITL"]
                val_splits = {"Tufts": 0.1, "HITL": 0.2}
            dataset_list, val_dataset_list, test_dataset_list = [], [], []
            for clas, name in zip(dataset_classes, dataset_names):
                dataset_kwargs: SimpleDatasetKwargs = {
                    **base_kwargs,
                    "dataset_name": name,
                }
                if name in train_names:
                    dataset_list.append((clas, dataset_kwargs))
                elif name in val_names:
                    val_split = val_splits[name]
                    val_kwargs: SimpleDatasetKwargs = {
                        **dataset_kwargs,
                        "split_val_size": val_split,
                        "split_test_size": 1 - val_split,
                    }
                    val_dataset_list.append((clas, val_kwargs))
                    test_dataset_list.append((clas, val_kwargs))
                else:
                    test_kwargs: SimpleDatasetKwargs = {
                        **dataset_kwargs,
                        "split_test_size": 1,
                    }
                    test_dataset_list.append((clas, test_kwargs))
            all_dataset_lists: DatasetLists[SimpleDataset, SimpleDatasetKwargs] = {
                "dataset_list": dataset_list,
                "val_dataset_list": val_dataset_list,
                "test_dataset_list": test_dataset_list,
            }

        if self.dataset in ["all", "all2"]:
            return all_dataset_lists

        if ":" not in self.dataset:
            dataset_class = dataset_class_map[self.dataset]
            dataset_kwargs: SimpleDatasetKwargs = {
                **base_kwargs,
                "split_val_size": 0.15,
                "split_test_size": 0.15,
                "dataset_name": self.dataset,
            }
            return {"dataset_list": [(dataset_class, dataset_kwargs)]}

        if splitted_dataset[0] in ["all", "all2"]:
            _, test_name = splitted_dataset
            test_dataset_list = list(
                filter(
                    lambda x: x[1].get("dataset_name") == test_name,
                    all_dataset_lists["test_dataset_list"],
                )
            )
            all_dataset_lists["test_dataset_list"] = test_dataset_list
            return all_dataset_lists

        if len(splitted_dataset) == 2:
            dataset_name, test_name = splitted_dataset
            dataset_kwargs: SimpleDatasetKwargs = {
                **base_kwargs,
                "split_val_size": 0.2,
                "dataset_name": dataset_name,
            }
            dataset_class = dataset_class_map[dataset_name]
            dataset_lists: DatasetLists = {
                "dataset_list": [(dataset_class, dataset_kwargs)]
            }
        elif len(splitted_dataset) == 3:
            train_name, val_name, test_name = splitted_dataset
            train_kwargs: SimpleDatasetKwargs = {
                **base_kwargs,
                "split_val_size": 0,
                "dataset_name": train_name,
            }
            val_kwargs: SimpleDatasetKwargs = {
                **base_kwargs,
                "split_val_size": 1,
                "dataset_name": val_name,
            }
            train_class = dataset_class_map[train_name]
            val_class = dataset_class_map[val_name]
            dataset_lists: DatasetLists = {
                "dataset_list": [(train_class, train_kwargs)],
                "val_dataset_list": [(val_class, val_kwargs)],
            }

        if test_name == "":
            dataset_lists["test_dataset_list"] = []
            return dataset_lists
        test_kwargs: SimpleDatasetKwargs = {
            **base_kwargs,
            "split_test_size": 1,
            "dataset_name": test_name,
        }
        test_class = dataset_class_map[test_name]
        dataset_lists["test_dataset_list"] = [(test_class, test_kwargs)]
        return dataset_lists


class MetaRunner(Runner):
    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        config: ConfigMetaLearner = self.config  # type: ignore

        suggest_or_parse_model(config, optuna_trial, self.optuna_config)

        if optuna_trial is not None:
            important_config = suggest_basic(config, optuna_trial)
        else:
            important_config = parse_basic(config, self.optuna_config)
        important_config = {**self.get_model_config(), **important_config}

        bg_weight = (
            optuna_trial.suggest_float("bg_weight", 0.01, 1.0, log=True)
            if optuna_trial is not None
            else self.optuna_config.get("hyperparams", {}).get("bg_weight", 0.1)
        )
        config["model"]["bg_weight"] = bg_weight  # type: ignore
        important_config["bg_weight"] = bg_weight

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        config["sampler_params"] = {
            "n_startup_trials": 10,
            "n_ei_candidates": 30,
            "multivariate": True,
            "group": True,
            "constant_liar": True,
            "seed": self.seed,
        }
        config["pruner"] = "median"
        config["pruner_params"] = {
            "n_warmup_steps": 21,
            "interval_steps": 2,
            "n_min_trials": 3,
        }
        config["pruner_patience"] = 0
        if not self.dummy:
            config["num_folds"] = 2
            config["timeout_sec"] = 3 * 24 * 3600
        return config

    def make_dataset_lists(
        self, query_fold: int, dummy: bool
    ) -> DatasetLists[FewSparseDataset, FewSparseDatasetKwargs]:
        batch_size = self.config["data"]["batch_size"]
        image_size = self.config["data"]["resize_to"][0]

        if self.mode == "test":
            query_batch = 5
        elif "ori" in self.learner_type.split("-"):
            query_batch = batch_size
        else:
            query_batch = 10
        if image_size == 512:
            sparsity_params = teeth_sparsity_params_512
        elif image_size == 256:
            sparsity_params = teeth_sparsity_params_256
        else:
            raise ValueError(f"No predefined params for image size {image_size}")
        base_kwargs: FewSparseDatasetKwargs = {
            "seed": self.seed,
            "split_val_fold": 0,
            "split_test_fold": 0,
            "cache_data": True,
            "support_query_data": "split",
            "query_batch_size": query_batch,
            "split_query_size": 0.5,
            "split_query_fold": query_fold,
            "sparsity_params": sparsity_params,
        }

        if dummy:
            dummy_kwargs: FewSparseDatasetKwargs = {
                "size": 4,
                "shot_options": (1, 3),
                "support_batch_mode": "mixed",
                "query_batch_size": 2,
                "num_iterations": 2,
            }
        else:
            dummy_kwargs = {}

        if "ori" in self.learner_type.split("-"):
            train_kwargs: FewSparseDatasetKwargs = {
                "shot_options": batch_size,
                "sparsity_options": [("random", "random")],
                "support_batch_mode": "mixed",
                "num_iterations": 5,
            }
        else:
            train_kwargs: FewSparseDatasetKwargs = {
                "shot_options": (1, 20),
                "sparsity_options": [
                    ("point", (5, 50)),
                    ("grid", (0.1, 1.0)),
                    ("contour", (0.1, 1.0)),
                    ("skeleton", (0.1, 1.0)),
                ],
                "support_batch_mode": "mixed",
                "num_iterations": 3.0,
            }

        val_kwargs: FewSparseDatasetKwargs = {
            "shot_options": [5, 10, 15],
            "sparsity_options": [
                ("point", [13, 25, 37]),
                ("grid", [0.25, 0.5, 0.75]),
                ("contour", [0.25, 0.5, 0.75]),
                ("skeleton", [0.25, 0.5, 0.75]),
            ],
            "support_batch_mode": "permutation",
        }

        test_kwargs: FewSparseDatasetKwargs = {
            "shot_options": [1, 5, 10, 15, 20],
            "sparsity_options": [
                ("point", [5, 13, 25, 37, 50]),
                ("grid", [0.1, 0.25, 0.5, 0.75, 1.0]),
                ("contour", [0.1, 0.25, 0.5, 0.75, 1.0]),
                ("skeleton", [0.1, 0.25, 0.5, 0.75, 1.0]),
            ],
            "support_query_data": "mixed",
            "support_batch_mode": "full_permutation",
        }

        dataset_names = [
            "HITL",
            "Adnan-Umer",
            "Dual-Labeled",
            "Tufts",
            "UFBA-425",
        ]
        dataset_classes = [
            HITLFSDataset,
            AdnanUmerFSDataset,
            DualLabeledFSDataset,
            TuftsFSDataset,
            UFBA425FSDataset,
        ]

        splitted_dataset = self.dataset.split(":")

        if splitted_dataset[0] not in ["all", "all2"]:
            raise ValueError(
                f"Meta-learners only support dataset with 'all' or 'all2, got {self.dataset}"
            )

        if splitted_dataset[0] == "all":
            train_names = ["HITL"]
            val_names = ["Tufts", "Dual-Labeled"]
            val_splits = {"Tufts": 0.1, "Dual-Labeled": 0.2}
        else:
            train_names = ["Dual-Labeled"]
            val_names = ["Tufts", "HITL"]
            val_splits = {"Tufts": 0.1, "HITL": 0.2}

        dataset_list, val_dataset_list, test_dataset_list = [], [], []
        for clas, name in zip(dataset_classes, dataset_names):
            if name in train_names:
                hitl_kwargs: FewSparseDatasetKwargs = {
                    **base_kwargs,
                    **train_kwargs,
                    "dataset_name": name,
                    **dummy_kwargs,
                }
                dataset_list.append((clas, hitl_kwargs))
            elif name in val_names:
                val_split = val_splits[name]
                val_kwargs_specific: FewSparseDatasetKwargs = {
                    **base_kwargs,
                    **val_kwargs,
                    "dataset_name": name,
                    "split_val_size": val_split,
                    "split_test_size": 1 - val_split,
                    **dummy_kwargs,
                }
                val_dataset_list.append((clas, val_kwargs_specific))
            if name not in train_names:
                test_kwargs_specific: FewSparseDatasetKwargs = {
                    **base_kwargs,
                    **test_kwargs,
                    "dataset_name": name,
                    "split_test_size": 1,
                    **dummy_kwargs,
                }
                if name in val_names:
                    val_split = val_splits[name]
                    test_kwargs_specific["split_val_size"] = val_split
                    test_kwargs_specific["split_test_size"] = 1 - val_split
                test_dataset_list.append((clas, test_kwargs_specific))

        all_dataset_lists: DatasetLists[FewSparseDataset, FewSparseDatasetKwargs] = {
            "dataset_list": dataset_list,
            "val_dataset_list": val_dataset_list,
            "test_dataset_list": test_dataset_list,
        }

        if self.dataset in ["all", "all2"]:
            return all_dataset_lists

        _, test_name = splitted_dataset
        test_dataset_list = list(
            filter(
                lambda x: x[1].get("dataset_name") == test_name,
                all_dataset_lists["test_dataset_list"],
            )
        )
        all_dataset_lists["test_dataset_list"] = test_dataset_list
        return all_dataset_lists


class WeaselRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[WeaselLearner], WeaselLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)
        loss, metric = define_loss_metric(self.config)

        kwargs: WeaselLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": loss,
            "metric": metric,
            "optuna_trial": optuna_trial,
        }

        return WeaselLearner, kwargs

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        important_config = super().update_config(optuna_trial)
        config: ConfigWeasel = self.config  # type: ignore

        if optuna_trial is not None:
            ws_update_rate = optuna_trial.suggest_float("ws_update_rate", 0.1, 1.0)
            ws_tune_epochs = optuna_trial.suggest_int("ws_tune_epochs", 1, 40)
            config["weasel"]["update_param_rate"] = ws_update_rate
            important_config["ws_update_rate"] = ws_update_rate
            config["weasel"]["tune_epochs"] = ws_tune_epochs
            important_config["ws_tune_epochs"] = ws_tune_epochs
        else:
            hyperparams = self.optuna_config.get("hyperparams", {})
            ws_update_rate = hyperparams.get("ws_update_rate")
            ws_tune_epochs = hyperparams.get("ws_tune_epochs")
            if isinstance(ws_update_rate, float):
                config["weasel"]["update_param_rate"] = ws_update_rate
                important_config["ws_update_rate"] = ws_update_rate
            if isinstance(ws_tune_epochs, int):
                config["weasel"]["tune_epochs"] = ws_tune_epochs
                important_config["ws_tune_epochs"] = ws_tune_epochs

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        return config


class ProtosegRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[ProtosegLearner], ProtoSegLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)
        loss, metric = define_loss_metric(self.config)

        kwargs: ProtoSegLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": loss,
            "metric": metric,
            "optuna_trial": optuna_trial,
        }

        return ProtosegLearner, kwargs

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        important_config = super().update_config(optuna_trial)
        config: ConfigProtoSeg = self.config  # type: ignore

        if optuna_trial is not None:
            if self.learner_type == "PS-ori":
                ps_embedding = config["protoseg"]["embedding_size"]
            else:
                ps_embedding = optuna_trial.suggest_int("ps_embedding", 2, 16)
            config["protoseg"]["embedding_size"] = ps_embedding
            important_config["ps_embedding"] = ps_embedding
        else:
            hyperparams = self.optuna_config.get("hyperparams", {})
            ps_embedding = hyperparams.get("ps_embedding")
            if isinstance(ps_embedding, int):
                config["protoseg"]["embedding_size"] = ps_embedding
                important_config["ps_embedding"] = ps_embedding

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        return config


class PANetRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[PANetLearner], PANetLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)
        loss, metric = define_loss_metric(self.config)

        kwargs: PANetLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": loss,
            "metric": metric,
            "optuna_trial": optuna_trial,
        }

        return PANetLearner, kwargs

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        important_config = super().update_config(optuna_trial)
        config: ConfigPANet = self.config  # type: ignore

        if optuna_trial is not None:
            pa_embedding = optuna_trial.suggest_int("pa_embedding", 2, 16)
            pa_par_weight = optuna_trial.suggest_float("pa_par_weight", 0.0, 1.0)
            config["panet"]["embedding_size"] = pa_embedding
            important_config["pa_embedding"] = pa_embedding
            config["panet"]["par_weight"] = pa_par_weight
            important_config["pa_par_weight"] = pa_par_weight
        else:
            hyperparams = self.optuna_config.get("hyperparams", {})
            pa_embedding = hyperparams.get("pa_embedding")
            pa_par_weight = hyperparams.get("pa_par_weight")
            if isinstance(pa_embedding, int):
                config["panet"]["embedding_size"] = pa_embedding
                important_config["pa_embedding"] = pa_embedding
            if isinstance(pa_par_weight, float):
                config["panet"]["par_weight"] = pa_par_weight
                important_config["pa_par_weight"] = pa_par_weight

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        return config


class PASNetRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[PASNetLearner], PASNetLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)
        loss, metric = define_loss_metric(self.config)

        kwargs: PASNetLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": loss,
            "metric": metric,
            "optuna_trial": optuna_trial,
        }

        return PASNetLearner, kwargs

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        important_config = super().update_config(optuna_trial)
        config: ConfigPASNet = self.config  # type: ignore
        not_nc = "nc" not in self.learner_type.split("-")

        if optuna_trial is not None:
            pas_embedding = optuna_trial.suggest_int("pas_embedding", 2, 16)
            pas_par_weight = optuna_trial.suggest_float("pas_par_weight", 0.0, 1.0)
            pas_prototype_metric = optuna_trial.suggest_categorical(
                "pas_prototype_metric", ["cosine", "euclidean"]
            )
            pas_high_conf_thres = optuna_trial.suggest_float(
                "pas_high_conf_thres", 1 / self.config["data"]["num_classes"], 0.9
            )
            config["pasnet"]["embedding_size"] = pas_embedding
            important_config["pas_embedding"] = pas_embedding
            config["pasnet"]["par_weight"] = pas_par_weight
            important_config["pas_par_weight"] = pas_par_weight
            config["pasnet"]["prototype_metric_func"] = pas_prototype_metric  # type: ignore
            important_config["pas_prototype_metric"] = pas_prototype_metric
            config["pasnet"]["high_confidence_threshold"] = pas_high_conf_thres
            important_config["pas_high_conf_thres"] = pas_high_conf_thres
            if not_nc:
                pas_consistency_weight = optuna_trial.suggest_float(
                    "pas_consistency_weight", 0.0, 1.0
                )
                pas_consistency_metric = optuna_trial.suggest_categorical(
                    "pas_consistency_metric", ["cosine", "euclidean"]
                )
                config["pasnet"]["consistency_weight"] = pas_consistency_weight
                important_config["pas_consistency_weight"] = pas_consistency_weight
                config["pasnet"]["consistency_metric_func"] = pas_consistency_metric  # type: ignore
                important_config["pas_consistency_metric"] = pas_consistency_metric
        else:
            hyperparams = self.optuna_config.get("hyperparams", {})
            pas_embedding = hyperparams.get("pas_embedding")
            pas_par_weight = hyperparams.get("pas_par_weight")
            pas_prototype_metric = hyperparams.get("pas_prototype_metric")
            pas_high_conf_thres = hyperparams.get("pas_high_conf_thres")
            if isinstance(pas_embedding, int):
                config["pasnet"]["embedding_size"] = pas_embedding
                important_config["pas_embedding"] = pas_embedding
            if isinstance(pas_par_weight, float):
                config["pasnet"]["par_weight"] = pas_par_weight
                important_config["pas_par_weight"] = pas_par_weight
            if isinstance(pas_prototype_metric, str) and (
                pas_prototype_metric == "cosine" or pas_prototype_metric == "euclidean"
            ):
                config["pasnet"]["prototype_metric_func"] = pas_prototype_metric
                important_config["pas_prototype_metric"] = pas_prototype_metric
            if isinstance(pas_high_conf_thres, float):
                config["pasnet"]["high_confidence_threshold"] = pas_high_conf_thres
                important_config["pas_high_conf_thres"] = pas_high_conf_thres
            if not_nc:
                pas_consistency_weight = hyperparams.get("pas_consistency_weight")
                pas_consistency_metric = hyperparams.get("pas_consistency_metric")
                if isinstance(pas_consistency_weight, float):
                    config["pasnet"]["consistency_weight"] = pas_consistency_weight
                    important_config["pas_consistency_weight"] = pas_consistency_weight
                if isinstance(pas_consistency_metric, str) and (
                    pas_consistency_metric == "cosine"
                    or pas_consistency_metric == "euclidean"
                ):
                    config["pasnet"]["consistency_metric_func"] = pas_consistency_metric
                    important_config["pas_consistency_metric"] = pas_consistency_metric

        self.config = config
        return important_config

    def make_optuna_config(self) -> OptunaConfig:
        config = super().make_optuna_config()
        return config


def get_runner_class(learner: str) -> Type[Runner]:
    runner_name = learner.split("-")[0]
    if runner_name == "SL":
        return SimpleRunner
    elif runner_name == "WS":
        return WeaselRunner
    elif runner_name == "PS":
        return ProtosegRunner
    elif runner_name == "PA":
        return PANetRunner
    elif runner_name == "PAS":
        return PASNetRunner
    else:
        raise ValueError(f"Unknown runner: {runner_name}")
