from typing import Type

import optuna

from config.config_maker import gen_id
from config.config_type import (
    ConfigMetaLearner,
    ConfigPANet,
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigUnion,
    ConfigWeasel,
)
from config.optuna import OptunaConfig
from data.few_sparse_dataset import FewSparseDataset
from data.simple_dataset import SimpleDataset
from data.typings import FewSparseDatasetKwargs, SimpleDatasetKwargs
from learners.panet_learner import PANetLearner
from learners.panet_unet import PANetUnet
from learners.protoseg_learner import ProtosegLearner
from learners.protoseg_unet import ProtosegUnet
from learners.simple_learner import SimpleLearner
from learners.simple_unet import SimpleUnet
from learners.typings import (
    DatasetLists,
    PANetLearnerKwargs,
    ProtoSegLearnerKwargs,
    SimpleLearnerKwargs,
    WeaselLearnerKwargs,
)
from learners.weasel_learner import WeaselLearner
from learners.weasel_unet import WeaselUnet
from runners.runner import Runner
from tasks.optic_disc_cup.datasets import (
    DrishtiTestFSDataset,
    DrishtiTestSimpleDataset,
    DrishtiTrainFSDataset,
    DrishtiTrainSimpleDataset,
    RefugeTestFSDataset,
    RefugeTestSimpleDataset,
    RefugeTrainFSDataset,
    RefugeTrainSimpleDataset,
    RefugeValFSDataset,
    RefugeValSimpleDataset,
    RimOne3TestFSDataset,
    RimOne3TestSimpleDataset,
    RimOne3TrainFSDataset,
    RimOne3TrainSimpleDataset,
    drishti_sparsity_params,
    refuge_train_sparsity_params,
    refuge_val_test_sparsity_params,
    rim_one_3_sparsity_params,
)
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU


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
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
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
            "split_val_size": 0.2,
        }
        refuge_test_kwargs: SimpleDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            "dataset_name": "REFUGE-test",
            "split_test_size": 1,
        }

        test_mult = 2 if self.mode == "profile-test" else 1
        dataset_list = [
            (RefugeValSimpleDataset, refuge_val_kwargs),
            (RimOne3TrainSimpleDataset, rim_one_3_train_kwargs),
            (DrishtiTrainSimpleDataset, drishti_train_kwargs),
        ]
        test_dataset_list = [
            (RefugeTestSimpleDataset, refuge_test_kwargs),
            (RimOne3TestSimpleDataset, rim_one_3_test_kwargs),
            (DrishtiTestSimpleDataset, drishti_test_kwargs),
        ]
        if self.dataset.startswith("rt-"):
            dataset_rt = [(RefugeTrainSimpleDataset, refuge_train_kwargs)]
            if self.dataset == "rt-REF":
                dataset_list = dataset_list[:1]
                test_dataset_list = test_dataset_list[:1]
            elif self.dataset == "rt-REF-test":
                test_dataset_list = test_dataset_list[:1]
            elif self.dataset == "rt-RO3-test":
                test_dataset_list = test_dataset_list[1:2]
            elif self.dataset == "rt-DGS-test":
                test_dataset_list = test_dataset_list[2:]
            elif self.dataset != "rt-all":
                raise ValueError(f"Invalid dataset: {self.dataset}")
            return {
                "dataset_list": dataset_rt,
                "val_dataset_list": dataset_list,
                "test_dataset_list": test_dataset_list * test_mult,
            }
        if self.dataset == "REF":
            dataset_list = dataset_list[:1]
            test_dataset_list = test_dataset_list[:1]
        elif self.dataset == "RO3":
            dataset_list = dataset_list[1:2]
            test_dataset_list = test_dataset_list[1:2]
        elif self.dataset == "DGS":
            dataset_list = dataset_list[2:]
            test_dataset_list = test_dataset_list[2:]
        elif self.dataset == "REF-RO3":
            dataset_list = dataset_list[:1]
            test_dataset_list = test_dataset_list[1:2]
        elif self.dataset == "REF-DGS":
            dataset_list = dataset_list[:1]
            test_dataset_list = test_dataset_list[2:]
        elif self.dataset == "RO3-REF":
            dataset_list = dataset_list[1:2]
            test_dataset_list = test_dataset_list[:1]
        elif self.dataset == "RO3-DGS":
            dataset_list = dataset_list[1:2]
            test_dataset_list = test_dataset_list[2:]
        elif self.dataset == "DGS-REF":
            dataset_list = dataset_list[2:]
            test_dataset_list = test_dataset_list[:1]
        elif self.dataset == "DGS-RO3":
            dataset_list = dataset_list[2:]
            test_dataset_list = test_dataset_list[1:2]
        elif self.dataset == "REF-test":
            test_dataset_list = test_dataset_list[:1]
        elif self.dataset == "RO3-test":
            test_dataset_list = test_dataset_list[1:2]
        elif self.dataset == "DGS-test":
            test_dataset_list = test_dataset_list[2:]
        elif self.dataset != "all":
            raise ValueError(f"Invalid dataset: {self.dataset}")
        return {
            "dataset_list": dataset_list,
            "test_dataset_list": test_dataset_list * test_mult,
        }


class MetaRunner(Runner):
    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        config: ConfigMetaLearner = self.config  # type: ignore

        if optuna_trial is not None:
            important_config = suggest_basic(config, optuna_trial)
        else:
            important_config = parse_basic(
                config, self.optuna_config.get("hyperparams", {})
            )

        variable_max_batch = 16
        variable_epochs = 25
        homogen_batch = 10
        homogen_thresholds = (0.7, 0.8)
        homogen_count = 30
        homogen_epochs = 50
        test_shots = [1, 5, 10, 15, 20]
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
            batch_size = (self.number_of_multi // len(test_shots)) + 1
            shot_idx = self.number_of_multi % len(test_shots)
            config["data"]["batch_size"] = batch_size
            important_config["batch_size"] = batch_size
            important_config["shot"] = test_shots[shot_idx]
            if self.number_of_multi == (variable_max_batch * len(test_shots) - 1):
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
            "min_resource": 5,
            "max_resource": self.config["learn"]["num_epochs"],
            "reduction_factor": 2,
            "bootstrap_count": 2,
        }
        if not self.dummy:
            config["num_folds"] = 2
            config["timeout_sec"] = 24 * 3600
        return config

    def make_dataset_lists(
        self, query_fold: int, dummy: bool
    ) -> DatasetLists[FewSparseDataset, FewSparseDatasetKwargs]:
        if self.mode == "test":
            query_batch = 5
        elif self.mode == "profile-test":
            query_batch = self.config["data"]["batch_size"]
        elif "ori" in self.learner_type.split("-"):
            query_batch = self.config["data"]["batch_size"]
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
                "max_items": 4,
                "shot_options": self.config["data"]["batch_size"],
                "support_batch_mode": "mixed",
                "query_batch_size": self.config["data"]["batch_size"],
                "num_iterations": 2,
            }
        else:
            dummy_kwargs = {}

        if "ori" in self.learner_type.split("-"):
            train_kwargs: FewSparseDatasetKwargs = {
                "shot_options": self.config["data"]["batch_size"],
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

        if self.mode == "profile-test":
            shot_idx = self.number_of_multi % 5
            test_kwargs: FewSparseDatasetKwargs = {
                "shot_options": [1, 5, 10, 15, 20][shot_idx : shot_idx + 1],
                "sparsity_options": [("dense", [0])],
                "support_query_data": "mixed",
                "support_batch_mode": "full_permutation",
            }
        else:
            test_kwargs: FewSparseDatasetKwargs = {
                "shot_options": [1, 5, 10, 15, 20],
                "sparsity_options": [
                    ("point", [1, 13, 25, 37, 50]),
                    ("grid", [0.1, 0.25, 0.5, 0.75, 1.0]),
                    ("contour", [0.1, 0.25, 0.5, 0.75, 1.0]),
                    ("skeleton", [0.1, 0.25, 0.5, 0.75, 1.0]),
                    ("region", [0.1, 0.25, 0.5, 0.75, 1.0]),
                ],
                "support_query_data": "mixed",
                "support_batch_mode": "full_permutation",
            }

        rim_one_3_train_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **val_kwargs,
            "dataset_name": "RIM-ONE-3-train",
            "split_val_size": 1,
            "sparsity_params": rim_one_3_sparsity_params,
            **dummy_kwargs,
        }
        rim_one_3_test_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **test_kwargs,
            "dataset_name": "RIM-ONE-3-test",
            "split_test_size": 1,
            "sparsity_params": rim_one_3_sparsity_params,
            **dummy_kwargs,
        }
        drishti_train_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **val_kwargs,
            "dataset_name": "DRISHTI-GS-train",
            "split_val_size": 1,
            "sparsity_params": drishti_sparsity_params,
            **dummy_kwargs,
        }
        if self.dataset == "all-B":
            drishti_train_kwargs["augment_flip"] = True
        drishti_test_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **test_kwargs,
            "dataset_name": "DRISHTI-GS-test",
            "split_test_size": 1,
            "sparsity_params": drishti_sparsity_params,
            **dummy_kwargs,
        }
        refuge_train_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **train_kwargs,
            "dataset_name": "REFUGE-train",
            "sparsity_params": refuge_train_sparsity_params,
            **dummy_kwargs,
        }
        refuge_val_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **val_kwargs,
            "dataset_name": "REFUGE-val",
            "split_val_size": 1,
            "sparsity_params": refuge_val_test_sparsity_params,
            **dummy_kwargs,
        }
        if self.dataset == "all-B":
            refuge_val_kwargs["max_items"] = 100
        refuge_test_kwargs: FewSparseDatasetKwargs = {  # noqa: F841
            **base_kwargs,
            **test_kwargs,
            "dataset_name": "REFUGE-test",
            "split_test_size": 1,
            "sparsity_params": refuge_val_test_sparsity_params,
            **dummy_kwargs,
        }

        dataset_list = [
            (RefugeTrainFSDataset, refuge_train_kwargs),
        ]
        val_dataset_list = [
            (RefugeValFSDataset, refuge_val_kwargs),
            (RimOne3TrainFSDataset, rim_one_3_train_kwargs),
            (DrishtiTrainFSDataset, drishti_train_kwargs),
        ]
        test_dataset_list = [
            (RefugeTestFSDataset, refuge_test_kwargs),
            (RimOne3TestFSDataset, rim_one_3_test_kwargs),
            (DrishtiTestFSDataset, drishti_test_kwargs),
        ]
        if self.dataset == "REF":
            val_dataset_list = val_dataset_list[:1]
            test_dataset_list = test_dataset_list[:1]
        elif self.dataset == "REF-test":
            test_dataset_list = test_dataset_list[:1]
        elif self.dataset == "RO3-test":
            test_dataset_list = test_dataset_list[1:2]
        elif self.dataset == "DGS-test":
            test_dataset_list = test_dataset_list[2:]
        elif self.dataset not in ["all", "all-B"]:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        return {
            "dataset_list": dataset_list,
            "val_dataset_list": val_dataset_list,
            "test_dataset_list": test_dataset_list,
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
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }

        return WeaselUnet, kwargs

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
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }

        return ProtosegUnet, kwargs

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
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }

        return PANetUnet, kwargs

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
    else:
        raise ValueError(f"Unknown runner: {runner_name}")
