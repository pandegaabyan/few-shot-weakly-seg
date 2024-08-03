import os

from dotenv import load_dotenv

import wandb
from config.constants import FILENAMES, WANDB_SETTINGS
from utils.logging import split_path
from utils.time import convert_epoch_to_iso_timestamp, convert_local_iso_to_utc_iso
from wandb.sdk.wandb_run import Run


def wandb_login():
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY is not set")
    wandb.login(key=wandb_api_key)


def wandb_path(dummy: bool) -> str:
    return (
        WANDB_SETTINGS["entity"]
        + "/"
        + WANDB_SETTINGS["dummy_project" if dummy else "project"]
    )


def wandb_get_run_id_by_name(run_name: str, dummy: bool = False) -> str:
    run = wandb.Api().runs(wandb_path(dummy), {"display_name": run_name})
    if len(run) == 0:
        raise ValueError(f"Run {run_name} not found")
    return run[0].id


def wandb_get_runs(
    start_time: float | str | None = None,
    end_time: float | str | None = None,
    dummy: bool = False,
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
    return wandb.Api().runs(wandb_path(dummy), filters=filter_dict)


def prepare_artifact_name(exp_name: str, run_name: str, suffix: str) -> str:
    run_name = run_name.replace("-", "").replace(" ", "-")
    return f"{exp_name}-{run_name}-{suffix}"


def prepare_ckpt_artifact_name(exp_name: str, run_name: str) -> str:
    return prepare_artifact_name(exp_name, run_name, "ckpt")


def prepare_ckpt_artifact_alias(ckpt_name: str) -> str:
    return ckpt_name.replace(" ", "-").replace("=", "_").removesuffix(".ckpt")


def prepare_study_ref_artifact_name(study_id: str) -> str:
    return f"{study_id}-study-ref"


def prepare_study_ckpt_artifact_name(study_id: str) -> str:
    return f"{study_id}-study-ckpt"


def wandb_delete_file(
    name: str,
    type: str,
    excluded_aliases: list[str] | None = None,
    dummy: bool = False,
):
    arts = wandb.Api().artifacts(type, f"{wandb_path(dummy)}/{name}")
    for art in arts:
        art: wandb.Artifact = art
        if len(art.aliases) == 0:
            art.delete(False)
        else:
            for alias in art.aliases:
                if excluded_aliases is None or alias not in excluded_aliases:
                    art.delete(True)


def wandb_log_file(
    run: Run | None,
    name: str,
    path: str,
    type: str,
    aliases: list[str] = ["latest"],
) -> wandb.Artifact | None:
    if run is None:
        return
    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(path)
    run.log_artifact(artifact, aliases=aliases)
    return artifact


def wandb_download_file(name: str, root: str, type: str, dummy: bool = False):
    artifact: wandb.Artifact = wandb.Api().artifact(
        f"{wandb_path(dummy)}/{name}", type=type
    )
    artifact.download(root)


def wandb_use_and_download_file(run: Run | None, name: str, root: str, type: str):
    if run is None:
        return
    artifact: wandb.Artifact = run.use_artifact(name, type=type)
    artifact.download(root)


def wandb_download_ckpt(ckpt_path: str):
    splitted_path = split_path(ckpt_path)
    ckpt_name = prepare_ckpt_artifact_name(*splitted_path[1:-1])
    ckpt_alias = prepare_ckpt_artifact_alias(splitted_path[-1])
    wandb_use_and_download_file(
        wandb.run,
        f"{ckpt_name}:{ckpt_alias}",
        os.path.split(ckpt_path)[0],
        "checkpoint",
    )


def wandb_download_config(exp_name: str, run_name: str):
    artifact_name = prepare_artifact_name(
        exp_name,
        run_name,
        "conf",
    )
    root = os.path.join(
        FILENAMES["log_folder"],
        exp_name,
        run_name,
    )
    wandb_use_and_download_file(
        wandb.run,
        artifact_name + ":base",
        root,
        "configuration",
    )
    wandb_use_and_download_file(
        wandb.run,
        artifact_name + ":latest",
        root,
        "configuration",
    )


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
