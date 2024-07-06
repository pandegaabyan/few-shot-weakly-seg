import os

from dotenv import load_dotenv

import wandb
from config.constants import WANDB_SETTINGS
from utils.time import convert_epoch_to_iso_timestamp, convert_local_iso_to_utc_iso


def wandb_login():
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY is not set")
    wandb.login(key=wandb_api_key)


def wandb_get_runs(
    start_time: float | str | None = None,
    end_time: float | str | None = None,
    dummy: bool = True,
):
    filter_dict = {}
    if start_time is not None:
        if isinstance(start_time, float):
            start_time = convert_epoch_to_iso_timestamp(start_time, True)
        elif isinstance(start_time, str):
            start_time = convert_local_iso_to_utc_iso(start_time)
        filter_dict.update({"created_at": {"$gte": start_time}})
    if end_time is not None:
        if isinstance(end_time, float):
            end_time = convert_epoch_to_iso_timestamp(end_time, True)
        elif isinstance(end_time, str):
            end_time = convert_local_iso_to_utc_iso(end_time)
        filter_dict.update({"created_at": {"$lte": end_time}})
    wandb_path = (
        WANDB_SETTINGS["entity"]
        + "/"
        + WANDB_SETTINGS["dummy_project" if dummy else "project"]
    )
    return wandb.Api().runs(wandb_path, filters=filter_dict)


def wandb_log_dataset_ref(dataset_path: str, dataset_name: str, dummy: bool = False):
    wandb_login()
    wandb.init(
        tags=["helper"],
        project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
        name=f"log dataset {dataset_name}",
    )
    dataset_artifact = wandb.Artifact(dataset_name, type="dataset")
    dataset_artifact.add_reference(f"file://{dataset_path}")
    wandb.log_artifact(dataset_artifact)
    wandb.finish()


def wandb_delete_old_tables(run_id: str | None, dummy: bool = False):
    if run_id is None:
        return
    wandb_path = (
        WANDB_SETTINGS["entity"]
        + "/"
        + WANDB_SETTINGS["dummy_project" if dummy else "project"]
        + "/"
        + run_id
    )
    run = wandb.Api().run(wandb_path)
    for artifact in run.logged_artifacts():
        if artifact.type == "run_table" and "latest" not in artifact.aliases:
            artifact.delete(delete_aliases=True)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]
