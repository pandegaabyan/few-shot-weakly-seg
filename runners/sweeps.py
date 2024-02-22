import os
from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

import wandb
from config.config_type import ConfigUnion
from config.constants import FILENAMES, WANDB_SETTINGS
from utils.logging import check_mkdir, dump_json, load_json
from utils.wandb import wandb_login


class SweepMetric(TypedDict):
    name: str
    goal: Literal["minimize", "maximize"]
    target: NotRequired[float]


class SweepConfigBase(TypedDict):
    method: Literal["random", "grid", "bayes"]
    metric: SweepMetric
    parameters: dict[str, dict[str, Any]]


class SweepConfigFull(SweepConfigBase):
    sweep_id: str
    use_cv: bool
    counts: list[int]


def get_sweep_config_path(exp_name: str, sweep_id: str) -> str:
    return os.path.join(
        FILENAMES["log_folder"],
        exp_name,
        FILENAMES["sweep_config"].replace(".json", f"_{sweep_id}.json"),
    )


def initialize_sweep(
    config: ConfigUnion,
    sweep_config: SweepConfigBase,
    dummy: bool = False,
    use_cv: bool = False,
    count: int = 3,
) -> SweepConfigFull:
    if config.get("wandb") is None:
        raise ValueError("sweep use wandb and need wandb config")

    wandb_login()

    sweep_id = config.get("wandb", {}).get("sweep_id")
    if sweep_id:
        prev_sweep_config_path = get_sweep_config_path(
            config["learn"]["exp_name"], sweep_id
        )
        prev_sweep_config = load_json(prev_sweep_config_path)
        assert isinstance(prev_sweep_config, dict)
        assert isinstance(prev_sweep_config["counts"], list)
        prev_sweep_config["counts"].append(count)
        dump_json(prev_sweep_config_path, prev_sweep_config)
        return prev_sweep_config  # type: ignore

    sweep_id = wandb.sweep(
        dict(sweep_config),
        project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
    )
    sweep_config_full: SweepConfigFull = {
        **sweep_config,
        "sweep_id": sweep_id,
        "use_cv": use_cv,
        "counts": [count],
    }

    sweep_config_path = get_sweep_config_path(config["learn"]["exp_name"], sweep_id)

    check_mkdir(os.path.split(sweep_config_path)[0])
    dump_json(sweep_config_path, dict(sweep_config_full))

    wandb.init(
        tags=["helper"],
        project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
        group=config["learn"]["exp_name"],
        name=f"init sweep {sweep_id}",
        job_type="sweep",
    )
    sweep_artifact = wandb.Artifact(f"sweep-{sweep_id}", type="config")
    sweep_artifact.add_file(sweep_config_path, FILENAMES["sweep_config"])
    wandb.log_artifact(sweep_artifact)
    wandb.finish()

    return sweep_config_full
