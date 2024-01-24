import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn import metrics
from torch import optim, nn

from config.config_type import AllConfig
from config.constants import FILENAMES
from learners.losses import CustomLoss
from learners.utils import check_mkdir, check_rmtree, get_gpu_memory, load_json, dump_json, \
    get_name_from_function, get_name_from_instance


class BaseLearner(ABC):

    def __init__(self,
                 net: nn.Module,
                 config: AllConfig,
                 calc_metrics: Callable[[list[NDArray], list[NDArray]], tuple[dict, str, str]] | None = None,
                 calc_loss: CustomLoss | None = None,
                 optimizer: optim.Optimizer | None = None,
                 scheduler: optim.lr_scheduler.LRScheduler | None = None):
        self.net = net
        self.calc_metrics = calc_metrics
        self.config = self.check_and_clean_config(config)

        self.output_path = os.path.join(FILENAMES['output_folder'], self.config['learn']['exp_name'])
        self.ckpt_path = os.path.join(FILENAMES['checkpoint_folder'], self.config['learn']['exp_name'])

        self.checkpoint = {}
        self.initial_gpu_percent = 0

        if calc_loss is None:
            self.calc_loss = CustomLoss()
        else:
            self.calc_loss = calc_loss

        if optimizer is None:
            self.optimizer = optim.Adam([
                {'params': net.parameters(), 'lr': self.config['optimizer']['lr']}
            ])
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                       self.config['scheduler']['step_size'])
        else:
            self.scheduler = scheduler

    @staticmethod
    @abstractmethod
    def set_used_config() -> list[str]:
        pass

    @abstractmethod
    def save_configuration(self, is_new: bool):
        pass

    @abstractmethod
    def learn_process(self, epoch: int):
        pass

    def learn(self):

        gpu_percent, gpu_total = self.initialize_gpu_usage()

        # Loading optimizer state in case of resuming training.
        if self.config['learn']['should_resume']:
            ok = self.check_output_and_ckpt_dir()
            if not ok:
                print('No data from previous learning')
                return

            self.read_checkpoint()
            self.load_net_and_optimizer()

            self.initialize_log()
            self.save_configuration(False)

            self.print_and_log('Resume learning ...', end='\n')
            curr_epoch = self.checkpoint['epoch'] + 1

        else:
            ok = self.clear_output_and_ckpt_dir()
            if not ok:
                print('Learning canceled')
                return

            self.create_output_and_ckpt_dir()

            self.initialize_log()
            self.save_configuration(True)

            self.print_and_log('Start learning ...', end="\n")
            curr_epoch = 1

        if self.config['learn']["use_gpu"]:
            self.print_and_log('Using GPU with total memory %dMiB, %.2f%% is already used' %
                               (gpu_total, gpu_percent))

        # Iterating over epochs.
        for epoch in range(curr_epoch, self.config['learn']['num_epochs'] + 1):
            self.learn_process(epoch)

        self.print_and_log('Finish learning ...')
        self.remove_log_handlers()

    def check_and_clean_config(self, ori_config: AllConfig) -> AllConfig:
        new_config = {}
        for key in self.set_used_config():
            new_config[key] = ori_config[key]  # type: ignore
        return new_config

    def initialize_gpu_usage(self) -> tuple[float, int]:
        gpu_percent, gpu_total = 0, 0
        if self.config['learn']["use_gpu"]:
            gpu_percent, gpu_total = get_gpu_memory()
            self.initial_gpu_percent = gpu_percent
            self.net = self.net.cuda()
        return gpu_percent, gpu_total

    def check_output_and_ckpt_dir(self) -> bool:
        return os.path.exists(self.output_path) and os.path.exists(self.ckpt_path)

    def create_output_and_ckpt_dir(self):
        check_mkdir(FILENAMES['output_folder'])
        check_mkdir(self.output_path)
        check_mkdir(FILENAMES['checkpoint_folder'])
        check_mkdir(self.ckpt_path)

    def clear_output_and_ckpt_dir(self) -> bool:
        output_ok = check_rmtree(self.output_path)
        ckpt_ok = check_rmtree(self.ckpt_path)
        return output_ok and ckpt_ok

    def initialize_log(self):
        logging.basicConfig(filename=os.path.join(self.output_path, FILENAMES['learn_log']),
                            encoding='utf-8',
                            level=logging.INFO,
                            format='%(asctime)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            force=True)

    @staticmethod
    def print_and_log(message: str, start: str = '', end: str = ''):
        print(start + message + end)
        logging.info(start.replace("\n", "") + message + end.replace("\n", ""))

    @staticmethod
    def log_error():
        logging.error("Exception:", exc_info=True, stack_info=True)

    @staticmethod
    def remove_log_handlers():
        logger = logging.getLogger()
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

    def write_to_csv(self, filename: str, fieldnames: list[str], row: dict):
        filename = os.path.join(self.output_path, filename)
        if os.path.isfile(filename):
            with open(filename, 'a', encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
        else:
            with open(filename, 'w', encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

    def read_checkpoint(self):
        self.checkpoint = load_json(os.path.join(self.ckpt_path, FILENAMES['checkpoint']))

    def update_checkpoint(self, data: dict):
        self.checkpoint.update(data)
        dump_json(os.path.join(self.ckpt_path, FILENAMES['checkpoint']), self.checkpoint)

    def get_optimization_data_dict(self) -> dict:
        optimizer_data = [
            {key: param_group[key] for key in filter(lambda x: x != 'params', param_group.keys())}
            for param_group in self.optimizer.__dict__['param_groups']
        ]
        scheduler_data = {
            key: self.scheduler.__dict__[key]
            for key in filter(lambda x: x != 'optimizer' and not x.startswith('_'), self.scheduler.__dict__.keys())
        }
        return {
            'metrics': None if self.calc_metrics is None else get_name_from_function(self.calc_metrics),
            'loss': get_name_from_instance(self.calc_loss),
            'loss_data': self.calc_loss.params(),
            'optimizer': get_name_from_instance(self.optimizer),
            'optimizer_data': optimizer_data,
            'scheduler': get_name_from_instance(self.scheduler),
            'scheduler_data': scheduler_data
        }

    def save_net_as_text(self):
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        net_text = '# of parameters: ' + str(n_params) + '\n\n' + str(self.net)
        with open(os.path.join(self.output_path, FILENAMES['net_text']), "w") as net_file:
            net_file.write(net_text)

    def calc_and_log_metrics(self, labels: list[NDArray], preds: list[NDArray], message: str = '',
                             start: str = '', end: str = '') -> dict:
        if self.calc_metrics is None:
            iou_mean = metrics.jaccard_score(np.concatenate(labels, axis=0).ravel(),
                                             np.concatenate(preds, axis=0).ravel(),
                                             average='macro')
            score = {'iou_mean': iou_mean}
            score_text = '%.2f' % (iou_mean * 100)
            name = 'Mean IoU score'
        else:
            score, score_text, name = self.calc_metrics(labels, preds)
        if message == '':
            full_message = f'{name}: {score_text}'
        else:
            full_message = f'{name} - {message}: {score_text}'
        self.print_and_log(full_message, start, end)
        return score

    def save_torch_dict(self, state_dict: dict, filename: str, epoch: int = 0):
        prefix = f'ep{epoch}_' if epoch != 0 else ''
        torch.save(state_dict, os.path.join(self.ckpt_path, prefix + filename))

    def load_torch_dict(self, filename: str, epoch: int = 0) -> dict:
        prefix = f'ep{epoch}_' if epoch != 0 else ''
        return torch.load(os.path.join(self.ckpt_path, prefix + filename))

    def save_net_and_optimizer(self, epoch: int = 0):
        self.save_torch_dict(self.net.state_dict(), FILENAMES['net_state'], epoch)
        self.save_torch_dict(self.optimizer.state_dict(), FILENAMES['optimizer_state'], epoch)

    def load_net_and_optimizer(self, epoch: int = 0):
        self.net.load_state_dict(self.load_torch_dict(FILENAMES['net_state'], epoch))
        self.optimizer.load_state_dict(self.load_torch_dict(FILENAMES['optimizer_state'], epoch))
