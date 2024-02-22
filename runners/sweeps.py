import os

import wandb
from config.config_type import ConfigUnion
from config.constants import FILENAMES, WANDB_SETTINGS
from utils.logging import check_mkdir, dump_json, load_json
from utils.wandb import wandb_login


def get_sweep_config_path(exp_name: str, sweep_id: str) -> str:
    return os.path.join(
        FILENAMES["log_folder"],
        exp_name,
        FILENAMES["sweep_config"].replace(".json", f"_{sweep_id}.json"),
    )


def initialize_sweep(
    config: ConfigUnion, sweep_config: dict, dummy: bool = False
) -> dict:
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
        return prev_sweep_config

    clean_sweep_config = sweep_config.copy()
    clean_sweep_config.pop("count_per_agent")
    clean_sweep_config.pop("use_cv")
    sweep_id = wandb.sweep(
        clean_sweep_config,
        project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
    )
    sweep_config["sweep_id"] = sweep_id

    sweep_config_path = get_sweep_config_path(config["learn"]["exp_name"], sweep_id)

    check_mkdir(os.path.split(sweep_config_path)[0])
    dump_json(sweep_config_path, sweep_config)

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

    return sweep_config
