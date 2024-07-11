from typing import Any, Literal

from config_type import OptunaPruner, OptunaSampler
from typing_extensions import NotRequired, TypedDict


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


default_optuna_config: OptunaConfig = {
    "study_name": "",
    "direction": "maximize",
    "sampler": "tpe",
    "pruner": "hyperband",
    "num_folds": 1,
    "timeout_sec": 600,
    "sampler_params": {},
    "pruner_params": {},
}
