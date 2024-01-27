from typing import Iterable


def check_mkdir(dir_name: str):
    import os

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_rmtree(dir_name: str) -> bool:
    import os
    import shutil

    if os.path.exists(dir_name):
        user_confirm = input(
            f"directory {dir_name} is not empty, clear it and continue? (Y/N)"
        )
        if user_confirm != "Y":
            return False
        shutil.rmtree(dir_name)

    return True


def cycle_iterable(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def get_name_from_function(func: callable) -> str:
    return f"<function {func.__module__}.{func.__name__}>"


def get_name_from_instance(instance: object) -> str:
    return str(instance.__class__).replace("'", "").replace("class", "object")


def get_gpu_memory() -> tuple[float, int]:
    import re
    import subprocess

    command = "nvidia-smi"
    nvidia_smi_text = subprocess.check_output(command)
    memories = re.findall(r"\b\d+MiB", str(nvidia_smi_text))
    used_memory, total_memory = memories[0], memories[1]
    used_memory = int(used_memory[:-3])
    total_memory = int(total_memory[:-3])
    percent_memory = used_memory * 100 / total_memory
    return percent_memory, total_memory


def load_json(path: str) -> dict:
    import json

    with open(path, "r") as f:
        data = json.load(f)
    return data


def dump_json(path: str, data: dict):
    import json

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def add_suffix_to_filename(filename: str, suffix: str) -> str:
    return f"{suffix}.".join(filename.rsplit(".", 1))


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
        simple_stack = filename + ":" + str(item.lineno) + " " + item.line
        simple_stack_list.append(simple_stack)
    return simple_stack_list
