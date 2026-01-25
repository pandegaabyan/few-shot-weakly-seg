from typing import Any, Literal, Type

import optuna
from typing_extensions import TypedDict

OptunaSampler = Literal["random", "tpe", "cmaes", "qmc", "gp", "botorch"]
OptunaPruner = Literal["none", "median", "percentile", "asha", "hyperband", "threshold"]


class OptunaConfig(TypedDict, total=False):
    study_name: str
    direction: Literal["minimize", "maximize"]
    sampler: OptunaSampler
    pruner: OptunaPruner
    num_folds: int
    num_trials: int
    timeout_sec: int
    sampler_params: dict[str, Any]
    pruner_params: dict[str, Any]
    pruner_patience: int
    hyperparams: dict[str, bool | int | float | str]


sampler_classes: dict[OptunaSampler, Type[optuna.samplers.BaseSampler]] = {
    "random": optuna.samplers.RandomSampler,
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
    "gp": optuna.samplers.GPSampler,
    "botorch": optuna.integration.BoTorchSampler,
}
pruner_classes: dict[OptunaPruner, Type[optuna.pruners.BasePruner]] = {
    "none": optuna.pruners.NopPruner,
    "median": optuna.pruners.MedianPruner,
    "percentile": optuna.pruners.PercentilePruner,
    "asha": optuna.pruners.SuccessiveHalvingPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "threshold": optuna.pruners.ThresholdPruner,
}


default_optuna_config: OptunaConfig = {
    "study_name": "",
    "direction": "maximize",
    "sampler": "tpe",
    "pruner": "hyperband",
    "num_folds": 1,
    "sampler_params": {},
    "pruner_params": {},
    "hyperparams": {},
}
