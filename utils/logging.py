from typing import Any, Callable, Iterable

from utils.utils import convert_iso_timestamp_to_epoch


def get_name_from_function(func: Callable) -> str:
    return f"<function {func.__module__}.{func.__name__}>"


def get_name_from_class(cls: object) -> str:
    return str(cls).replace("'", "")


def get_name_from_instance(instance: object) -> str:
    return str(instance.__class__).replace("'", "").replace("class", "object")


def get_short_git_hash() -> str:
    import subprocess

    short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    short_hash = str(short_hash, "utf-8").strip()
    return short_hash


def get_simple_stack_list(start: int = 0, end: int | None = None) -> list[str]:
    import os
    import traceback

    simple_stack_list = []
    for item in traceback.extract_stack()[start:end]:
        try:
            filename = os.path.relpath(item.filename, os.getcwd())
        except ValueError:
            filename = item.filename
        simple_stack = (
            filename + ":" + str(item.lineno or "") + " " + (item.line or "(None)")
        )
        simple_stack_list.append(simple_stack)
    return simple_stack_list


def get_count_as_text(data: Iterable) -> str:
    from collections import Counter

    counted = Counter(data)
    return " ".join([f"{key}({count})" for key, count in counted.items()])


# def prepare_ckpt_path_for_artifact(ckpt_path: str) -> str:
#     return (
#         ckpt_path.replace("-", "").replace(" ", "-").replace("/", "-").replace("=", "_")
#     )
def prepare_ckpt_path_for_artifact(ckpt_path: str) -> tuple[str, str]:
    exp_name, run_name, ckpt_name = ckpt_path.split("/")
    run_name = run_name.replace("-", "").replace(" ", "-")
    ckpt_name = ckpt_name.replace(" ", "-").replace("=", "_").removesuffix(".ckpt")
    return f"{exp_name}-{run_name}", ckpt_name


def check_mkdir(dir_name: str):  # Error Here
    import os

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def check_rmtree(dir_name: str, force: bool = False) -> bool:
    import os
    import shutil

    if os.path.exists(dir_name):
        if not force:
            user_confirm = input(
                f"directory {dir_name} is not empty, clear it and continue? (Y/N) "
            )
            if user_confirm != "Y":
                return False
        shutil.rmtree(dir_name)

    return True


def load_json(path: str) -> dict | list:
    import json

    with open(path, "r") as f:
        data = json.load(f)
    return data


def dump_json(path: str, data: dict | list):
    import json

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def write_to_csv(filename: str, row: list[tuple[str, Any]]):
    import csv
    import os

    fieldnames = [r[0] for r in row]
    rowdict = dict(row)
    if os.path.isfile(filename):
        with open(filename, "a", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(rowdict)
    else:
        with open(filename, "w", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(rowdict)


def get_configuration(exp_name: str = "", run_name: str = "") -> dict:
    import os

    from config.constants import FILENAMES

    configuration_path = os.path.join(
        FILENAMES["log_folder"],
        exp_name,
        run_name,
        FILENAMES["configuration"],
    )
    config = load_json(configuration_path)
    assert isinstance(config, dict)
    return config


def get_full_ckpt_path(*paths: str, extension: str = ".ckpt") -> str:
    import os

    from config.constants import FILENAMES

    path = os.path.join(FILENAMES["checkpoint_folder"], *paths)
    if not path.endswith(extension):
        path += extension
    return path


def get_run_paths(
    start_time: float | str | None = None,
    end_time: float | str | None = None,
    dummy_only: bool = True,
) -> list[str]:
    import os

    from config.constants import FILENAMES

    if start_time is not None:
        if isinstance(start_time, str):
            start_time = convert_iso_timestamp_to_epoch(start_time)
    else:
        start_time = None

    if end_time is not None:
        if isinstance(end_time, str):
            end_time = convert_iso_timestamp_to_epoch(end_time)
    else:
        end_time = None

    run_list = []
    parent_path = FILENAMES["log_folder"]
    for exp_name in os.listdir(parent_path):
        exp_path = os.path.join(parent_path, exp_name)
        for run_name in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_name)
            run_time = os.path.getmtime(run_path)
            if start_time is not None and run_time < start_time:
                continue
            if end_time is not None and run_time > end_time:
                continue
            dummy_file_path = os.path.join(run_path, FILENAMES["dummy_file"])
            if dummy_only and not os.path.isfile(dummy_file_path):
                continue
            run_list.append(exp_name + "/" + run_name)

    return run_list