import os
from contextlib import contextmanager

import optuna
from dotenv import load_dotenv

import wandb
from config.constants import FILENAMES, WANDB_ENTITY
from utils.time import convert_epoch_to_iso_timestamp, convert_local_iso_to_utc_iso
from wandb.sdk.wandb_run import Run


def wandb_login():
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY is not set")
    wandb.login(key=wandb_api_key)


def get_wandb_project(dummy: bool = False) -> str:
    if dummy:
        return "few-shot-weakly-seg-dummy"
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_PROJECT")
    if not wandb_api_key:
        raise ValueError("WANDB_PROJECT is not set")
    return wandb_api_key


def wandb_path(dummy: bool) -> str:
    return WANDB_ENTITY + "/" + get_wandb_project(dummy)


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
    exp_name = exp_name.replace(" ", "-")
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


def wandb_delete_files(
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


def wandb_download_file(name: str, root: str, type: str, dummy: bool = False) -> str:
    artifact: wandb.Artifact = wandb.Api().artifact(
        f"{wandb_path(dummy)}/{name}", type=type
    )
    return str(artifact.file(root))


def wandb_use_and_download_file(
    run: Run | None, name: str, root: str, type: str
) -> str:
    if run is None:
        return ""
    artifact: wandb.Artifact = run.use_artifact(name, type=type)
    return str(artifact.file(root))


def wandb_download_ckpt(
    name: str,
    log_path: str,
    alias: str | None = None,
    study: bool = False,
    dummy: bool = False,
) -> str:
    type_name = "study-checkpoint" if study else "checkpoint"
    artifact_path = f"{wandb_path(dummy)}/{name}"
    if alias is None:
        alias = "latest"
    if alias in ["max", "min"]:
        arts = wandb.Api().artifacts(type_name, artifact_path)
        aliases = [alias for art in arts for alias in art.aliases]
        index = -1 if alias == "max" else 0
        alias = sorted(filter(lambda x: len(x) > 6, aliases))[index]
    assert alias is not None
    if study:
        if len(alias) < 6:
            all_alias = wandb.Api().artifact(artifact_path, type=type_name).aliases
            alias = sorted(filter(lambda x: len(x) > 6, all_alias))[0]
        if "fold" in alias:
            fold = alias[alias.index("fold_") + 5 :].split("-")[0]
            if fold != "0":
                log_path += f" F{fold}"
    artifact = f"{name}:{alias}"
    if wandb.run is None:
        return wandb_download_file(artifact, log_path, type_name, dummy)
    return wandb_use_and_download_file(wandb.run, artifact, log_path, type_name)


def wandb_download_config(exp_name: str, run_name: str) -> tuple[str, str]:
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
    base_config_path = wandb_use_and_download_file(
        wandb.run,
        artifact_name + ":base",
        root,
        "configuration",
    )
    latest_config_path = wandb_use_and_download_file(
        wandb.run,
        artifact_name + ":latest",
        root,
        "configuration",
    )
    return base_config_path, latest_config_path


def wandb_log_dataset_ref(dataset_path: str, dataset_name: str, dummy: bool = False):
    wandb_login()
    wandb.init(
        tags=["helper"],
        project=get_wandb_project(dummy),
        name=f"log dataset {dataset_name}",
        settings=wandb.Settings(_disable_stats=True),
    )
    dataset_artifact = wandb.Artifact(dataset_name, type="dataset")
    dataset_artifact.add_reference(f"file://{dataset_path}")
    wandb.log_artifact(dataset_artifact)
    wandb.finish()


@contextmanager
def wandb_use_alert():
    try:
        yield
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        if wandb.run is not None:
            wandb.run.alert(
                title=f"Error: {e}",
                text=f"ID: {wandb.run.id} | Group: {wandb.run.group or '-'} | Job Type: {wandb.run.job_type or '-'}",
                level=wandb.AlertLevel.ERROR,
            )
        raise
