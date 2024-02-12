import os
import time
from typing import Literal

from config.constants import FILENAMES, WANDB_SETTINGS
from utils.logging import check_rmtree, get_run_paths
from utils.utils import convert_iso_timestamp_to_epoch
from utils.wandb import wandb_get_runs


def clean_logging_data(
    after: str | int,
    dummy_only: bool = True,
    target: Literal["local", "wandb", "both"] = "both",
    force_clean: bool = False,
):
    if isinstance(after, int):
        current_epoch = time.time()
        start_time = current_epoch - after * 60
    else:
        start_time = after

    local_run_paths = []
    if target in ["local", "both"]:
        local_run_paths = get_run_paths(start_time=start_time, dummy_only=dummy_only)

    wandb_runs = []
    wandb_runs_dummy = []
    if target in ["wandb", "both"]:
        wandb_runs_dummy = wandb_get_runs(start_time=start_time, dummy=True)
        if not dummy_only:
            wandb_runs = wandb_get_runs(start_time=start_time, dummy=False)
    wandb_run_paths = [
        (run.group or "") + "/" + run.name for run in wandb_runs + wandb_runs_dummy
    ]

    combined_run_paths = set(local_run_paths + wandb_run_paths)
    annotated_run_paths = []
    for run in combined_run_paths:
        annotated_run = run + " "
        if run in local_run_paths:
            annotated_run += "(local)"
        if run in wandb_run_paths:
            annotated_run += "(wandb)"
        annotated_run_paths.append(annotated_run)

    print("The following runs will be cleaned:")
    for run in annotated_run_paths:
        print(run)
    if not force_clean:
        user_confirm = input("continue? (Y/N) ")
        if user_confirm != "Y":
            return

    parent_dirs = [
        FILENAMES["checkpoint_folder"],
        FILENAMES["tensorboard_folder"],
        FILENAMES["log_folder"],
    ]
    for parent in parent_dirs:
        for run in combined_run_paths:
            run_path = os.path.join(parent, run)
            if os.path.isfile(run_path):
                os.remove(run_path)
            else:
                check_rmtree(run_path, True)

    for run in wandb_runs:
        run.delete(delete_artifacts=True)
    for run in wandb_runs_dummy:
        run.delete(delete_artifacts=True)


def clean_local_wandb(before: str | int, force_clean: bool = False):
    if isinstance(before, int):
        current_epoch = time.time()
        end_time = current_epoch - before * 60
    else:
        end_time = convert_iso_timestamp_to_epoch(before)

    dirs_to_clean = []
    for dir in os.listdir(WANDB_SETTINGS["dir"]):
        if dir.endswith(".log"):
            continue
        dir_path = os.path.join(WANDB_SETTINGS["dir"], dir)
        if os.path.getmtime(dir_path) < end_time:
            dirs_to_clean.append(dir)

    print("The following folders will be cleaned from local wandb:")
    for dir in dirs_to_clean:
        print(dir)
    if not force_clean:
        user_confirm = input("continue? (Y/N) ")
        if user_confirm != "Y":
            return

    for dir in dirs_to_clean:
        check_rmtree(os.path.join(WANDB_SETTINGS["dir"], dir), True)
