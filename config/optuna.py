import os
from typing import Any, Literal, Type, TypedDict

import optuna
from dotenv import load_dotenv

OptunaSampler = Literal["random", "tpe", "cmaes", "qmc", "gp", "botorch"]


class OptunaConfig(TypedDict):
    study_name: str
    direction: Literal["minimize", "maximize"]
    sampler: OptunaSampler
    timeout_sec: int
    sampler_params: dict[str, Any]


sampler_classes: dict[OptunaSampler, Type[optuna.samplers.BaseSampler]] = {
    "random": optuna.samplers.RandomSampler,
    "tpe": optuna.samplers.TPESampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
    "gp": optuna.samplers.GPSampler,
    "botorch": optuna.integration.BoTorchSampler,
}


def get_optuna_storage(dummy: bool = False) -> optuna.storages.BaseStorage:
    if dummy:
        dir = "outputs"
        os.makedirs(dir, exist_ok=True)
        db_url = f"sqlite:///{dir}/optuna_dummy.sqlite3"
    else:
        load_dotenv()
        db_url = os.getenv("OPTUNA_DB_URL")
        if not db_url:
            raise ValueError("OPTUNA_DB_URL is not set")
    return optuna.storages.RDBStorage(url=db_url)
