from typing import Any, Type

import optuna

from config.config_maker import gen_id
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
from learners.metrics import BinaryIoUMetric
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
from tasks.skin_lesion.datasets import (
    ISIC16MELFSDataset,
    ISIC16MELSimpleDataset,
    ISIC1617NVFSDataset,
    isic1617_sparsity_params,
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
        model = optuna_config.get("hyperparams", {}).get("model")
        if isinstance(model, str):
            model_split = model.split("_", 2)
            config["model"]["arch"] = model_split[0]
            if len(model_split) == 2:
                config["model"]["backbone"] = model_split[1]


def define_loss(
    config: ConfigUnion,
) -> tuple[Type[CustomLoss], dict[str, Any]]:
    bg_weight = config["model"].get("bg_weight", 0.1)
    loss = (
        CustomLoss,
        {
            "mode": "bce",
            "ce_weights": [1.0 / bg_weight],
        },
    )
    return loss


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
            "loss": define_loss(self.config),
            "metric": (BinaryIoUMetric, {}),
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
        important_config = {"model": self.get_model_name(), **important_config}

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
            base_kwargs["size"] = 6

        isic16_mel_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "ISIC16-MEL",
            "split_val_size": 0.2,
        }

        return {
            "dataset_list": [(ISIC16MELSimpleDataset, isic16_mel_kwargs)],
            "test_dataset_list": [],
        }


class MetaRunner(Runner):
    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        config: ConfigMetaLearner = self.config  # type: ignore

        suggest_or_parse_model(config, optuna_trial, self.optuna_config)

        if optuna_trial is not None:
            important_config = suggest_basic(config, optuna_trial)
        else:
            important_config = parse_basic(config, self.optuna_config)
        important_config = {"model": self.get_model_name(), **important_config}

        bg_weight = (
            optuna_trial.suggest_float("bg_weight", 0.1, 1.0, log=True)
            if optuna_trial is not None
            else self.optuna_config.get("hyperparams", {}).get("bg_weight", 0.1)
        )
        config["model"]["bg_weight"] = bg_weight  # type: ignore
        important_config["bg_weight"] = bg_weight

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
            "min_resource": 5,
            "max_resource": self.config["learn"]["num_epochs"],
            "reduction_factor": 2,
            "bootstrap_count": 2,
        }
        if not self.dummy:
            config["num_folds"] = 2
            config["timeout_sec"] = 3 * 24 * 3600
        return config

    def make_dataset_lists(
        self, query_fold: int, dummy: bool
    ) -> DatasetLists[FewSparseDataset, FewSparseDatasetKwargs]:
        batch_size = self.config["data"]["batch_size"]

        if self.mode == "test":
            query_batch = 5
        elif "ori" in self.learner_type.split("-"):
            query_batch = batch_size
        else:
            query_batch = 10
        base_kwargs: FewSparseDatasetKwargs = {
            "seed": self.seed,
            "split_val_fold": 0,
            "split_test_fold": 0,
            "cache_data": True,
            "support_query_data": "split",
            "query_batch_size": query_batch,
            "split_query_size": 0.5,
            "split_query_fold": query_fold,
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
                    ("region", (0.1, 1.0)),
                ],
                "support_batch_mode": "mixed",
                "num_iterations": 5.0,
            }

        val_kwargs: FewSparseDatasetKwargs = {
            "shot_options": [5, 10, 15],
            "sparsity_options": [
                ("point", [13, 25, 37]),
                ("grid", [0.25, 0.5, 0.75]),
                ("contour", [0.25, 0.5, 0.75]),
                ("skeleton", [0.25, 0.5, 0.75]),
                ("region", [0.25, 0.5, 0.75]),
            ],
            "support_batch_mode": "permutation",
        }

        isic1617_nv_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **train_kwargs,
            "dataset_name": "ISIC1617-NV",
            "sparsity_params": isic1617_sparsity_params,
            **dummy_kwargs,
        }
        isic16_mel_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **val_kwargs,
            "dataset_name": "ISIC16-MEL",
            "split_val_size": 1,
            "sparsity_params": isic1617_sparsity_params,
            **dummy_kwargs,
        }

        return {
            "dataset_list": [(ISIC1617NVFSDataset, isic1617_nv_kwargs)],
            "val_dataset_list": [(ISIC16MELFSDataset, isic16_mel_kwargs)],
            "test_dataset_list": [],
        }


class WeaselRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[WeaselLearner], WeaselLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)

        kwargs: WeaselLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": define_loss(self.config),
            "metric": (BinaryIoUMetric, {}),
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
        config["pruner_patience"] = 1
        return config


class ProtosegRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[ProtosegLearner], ProtoSegLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)

        kwargs: ProtoSegLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": define_loss(self.config),
            "metric": (BinaryIoUMetric, {}),
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
        config["pruner_patience"] = 3
        return config


class PANetRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[PANetLearner], PANetLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)

        kwargs: PANetLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": define_loss(self.config),
            "metric": (BinaryIoUMetric, {}),
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
        config["pruner_patience"] = 3
        return config


class PASNetRunner(MetaRunner):
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[PASNetLearner], PASNetLearnerKwargs]:
        dataset_lists = self.make_dataset_lists(dataset_fold, self.dummy)

        kwargs: PASNetLearnerKwargs = {
            **dataset_lists,
            "config": self.config,
            "loss": define_loss(self.config),
            "metric": (BinaryIoUMetric, {}),
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
        config["pruner_patience"] = 3
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
