import os

import wandb
from config.config_type import ConfigUnion
from config.constants import FILENAMES, WANDB_SETTINGS
from utils.logging import check_mkdir, dump_json
from utils.wandb import wandb_login


def initialize_sweep(config: ConfigUnion, sweep_config: dict, dummy: bool = False):
    if config.get("wandb") is None:
        raise ValueError("sweep use wandb and need wandb config")

    sweep_metric = config.get("wandb", {}).get("sweep_metric")
    if sweep_metric is None:
        raise ValueError("sweep need sweep_metric in wandb config")
    sweep_config["metric"] = {"name": sweep_metric[0], "goal": sweep_metric[1]}

    if sweep_config.get("method") is None or sweep_config.get("parameters") is None:
        raise ValueError("sweep need method and parameters in sweep_config")

    wandb_login()

    clean_sweep_config = sweep_config.copy()
    clean_sweep_config.pop("count")
    sweep_id = wandb.sweep(clean_sweep_config, project=WANDB_SETTINGS["project"])
    sweep_config["sweep_id"] = sweep_id

    sweep_config_path = os.path.join(
        FILENAMES["log_folder"],
        config["learn"]["exp_name"],
        FILENAMES["sweep_config"],
    )

    check_mkdir(os.path.split(sweep_config_path)[0])
    dump_json(sweep_config_path, sweep_config)

    wandb_tags = ["helper"]
    if dummy:
        wandb_tags.append("dummy")
    wandb.init(
        tags=wandb_tags,
        group=config["learn"]["exp_name"],
        name=f"init sweep {sweep_id}",
        job_type="sweep",
    )
    sweep_artifact = wandb.Artifact(f"sweep-{sweep_id}", type="config")
    sweep_artifact.add_file(sweep_config_path, FILENAMES["sweep_config"])
    wandb.log_artifact(sweep_artifact)
    wandb.finish()

    return sweep_config
