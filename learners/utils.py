import os
import re
import shutil
import subprocess
from typing import Iterable


def check_mkdir(dir_name: str):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_rmtree(dir_name: str) -> bool:
    if os.path.exists(dir_name):
        user_confirm = input(f"directory {dir_name} is not empty, clear it and continue? (Y/N)")
        if user_confirm != 'Y':
            return False
        shutil.rmtree(dir_name)

    return True


def cycle_iterable(iterable: Iterable):
    while True:
        for x in iterable:
            yield x


def get_gpu_memory() -> tuple[float, int]:
    command = 'nvidia-smi'
    nvidia_smi_text = subprocess.check_output(command)
    [used_memory, total_memory] = re.findall(r'\b\d+MiB', str(nvidia_smi_text))
    used_memory = int(used_memory[:-3])
    total_memory = int(total_memory[:-3])
    percent_memory = used_memory * 100 / total_memory
    return percent_memory, total_memory