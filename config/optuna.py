import os
from typing import Any, Literal, Type

import optuna
from dotenv import load_dotenv
from typing_extensions import NotRequired, TypedDict

from config.constants import FILENAMES
from utils.logging import check_mkdir

OptunaSampler = Literal["random", "tpe", "cmaes", "qmc", "gp"]
OptunaPruner = Literal["none", "median", "percentile", "asha", "hyperband", "threshold"]


class OptunaConfig(TypedDict):
    study_name: str
    direction: Literal["minimize", "maximize"]
    sampler: OptunaSampler
    pruner: OptunaPruner
    num_folds: NotRequired[int]
    num_trials: NotRequired[int]
    timeout_sec: NotRequired[int]
    sampler_params: NotRequired[dict[str, Any]]
    pruner_params: NotRequired[dict[str, Any]]


sampler_classes: dict[OptunaSampler, Type[optuna.samplers.BaseSampler]] = {
    "random": optuna.samplers.RandomSampler,
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
    "gp": optuna.samplers.GPSampler,
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
}


def get_optuna_storage(dummy: bool = False) -> optuna.storages.BaseStorage:
    if dummy:
        log_dir = FILENAMES["log_folder"]
        check_mkdir(log_dir)
        db_url = f"sqlite:///{log_dir}/optuna_dummy.sqlite3"
    else:
        load_dotenv()
        db_url = os.getenv("OPTUNA_DB_URL")
        if not db_url:
            raise ValueError("OPTUNA_DB_URL is not set")
    return optuna.storages.RDBStorage(url=db_url)
