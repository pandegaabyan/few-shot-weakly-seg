import os
import re
import shutil
import subprocess
from typing import Iterable, Union

from torch import optim

from data.dataset_loaders import DatasetLoaderParamSimple
from learners.losses import CustomLoss


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
    memories = re.findall(r'\b\d+MiB', str(nvidia_smi_text))
    used_memory, total_memory = memories[0], memories[1]
    used_memory = int(used_memory[:-3])
    total_memory = int(total_memory[:-3])
    percent_memory = used_memory * 100 / total_memory
    return percent_memory, total_memory


def serialize_loader_param(param: DatasetLoaderParamSimple) -> dict:
    new_param: dict = param.copy()
    new_param['dataset_class'] = str(param['dataset_class']).replace('\'', '')
    return new_param


def serialize_dataset_params(meta_params: list[DatasetLoaderParamSimple],
                             tune_param: DatasetLoaderParamSimple) -> dict:
    return {
        'meta': [serialize_loader_param(mp) for mp in meta_params],
        'tune': serialize_loader_param(tune_param)
    }


def get_name_from_function(func: callable) -> str:
    return f'<function {func.__module__}.{func.__name__}>'


def get_name_from_instance(instance: object) -> str:
    return str(instance.__class__).replace('\'', '').replace('class', 'object')


def serialize_optimization_data(calc_metrics: Union[callable, None], loss: CustomLoss, optimizer: optim.Optimizer,
                                scheduler: optim.lr_scheduler.LRScheduler) -> dict:
    optimizer_data = [
        {key: param_group[key] for key in filter(lambda x: x != 'params', param_group.keys())}
        for param_group in optimizer.__dict__['param_groups']
    ]
    scheduler_data = {
        key: scheduler.__dict__[key]
        for key in filter(lambda x: x != 'optimizer' and not x.startswith('_'), scheduler.__dict__.keys())
    }
    return {
        'metrics': None if calc_metrics is None else get_name_from_function(calc_metrics),
        'loss': get_name_from_instance(loss),
        'loss_data': loss.params(),
        'optimizer': get_name_from_instance(optimizer),
        'optimizer_data': optimizer_data,
        'scheduler': get_name_from_instance(scheduler),
        'scheduler_data': scheduler_data
    }
