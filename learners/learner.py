import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import io
from torch import optim
from torch.utils.data import DataLoader

from config.config_type import AllConfig
from config.constants import FILENAMES
from data.dataset_loaders import DatasetLoaderItem
from data.types import TensorDataItem
from learners.utils import check_mkdir, check_rmtree, cycle_iterable, get_gpu_memory
from torchmeta.modules import MetaModule


class MetaLearner(ABC):

    def __init__(self,
                 net: MetaModule,
                 config: AllConfig,
                 meta_loaders: list[DatasetLoaderItem],
                 tune_loaders: list[DatasetLoaderItem],
                 func_calc_metrics: Callable[[list[NDArray], list[NDArray]], tuple[dict, str, str]]):
        self.config = config
        self.meta_loaders = meta_loaders
        self.tune_loaders = tune_loaders
        self.func_calc_metrics = func_calc_metrics

        self.output_path = os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'])
        self.ckpt_path = os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'])

        self.checkpoint = {}

        if self.config['learn']["use_gpu"]:
            self.net = net.cuda()
        else:
            self.net = net

        self.meta_optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * self.config['learn']['optimizer_lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': self.config['learn']['optimizer_lr'],
             'weight_decay': self.config['learn']['optimizer_weight_decay']}
        ], betas=(self.config['learn']['optimizer_momentum'], 0.99))

        self.scheduler = optim.lr_scheduler.StepLR(self.meta_optimizer,
                                                   self.config['learn']['scheduler_step_size'],
                                                   gamma=self.config['learn']['scheduler_gamma'])

    @abstractmethod
    def meta_train_test_step(self, train_data: TensorDataItem, test_data: TensorDataItem) -> float:
        pass

    @abstractmethod
    def tune_train_test_process(self, epoch: int,
                                tune_loader: DatasetLoaderItem) -> tuple[list[NDArray], list[NDArray], list[str]]:
        pass

    def learn(self):

        # Loading optimizer state in case of resuming training.
        if self.config['learn']['should_resume']:
            ok = self.check_output_and_ckpt_dir()
            if not ok:
                print('No data from previous learning')
                return

            self.read_checkpoint()
            self.load_net_and_optimizer()

            self.initialize_log()
            self.save_config_and_net()

            self.print_and_log('Resume learning ...', end='\n')
            curr_epoch = self.checkpoint['epoch'] + 1

        else:
            ok = self.clear_output_and_ckpt_dir()
            if not ok:
                print('Learning canceled')
                return

            self.create_output_and_ckpt_dir()

            self.initialize_log()
            self.save_config_and_net()

            self.print_and_log('Start learning ...', end="\n")
            curr_epoch = 1

        if self.config['learn']["use_gpu"]:
            gpu_percentage, gpu_total = get_gpu_memory()
            self.print_and_log('Using GPU with total memory %dMiB, %.2f%% is already used' %
                               (gpu_total, gpu_percentage))

        # Iterating over epochs.
        for epoch in range(curr_epoch, self.config['learn']['num_epochs'] + 1):

            # Meta training on source datasets.
            self.meta_train_test(epoch)
            self.save_net_and_optimizer()
            self.update_checkpoint({'epoch': epoch})

            if epoch % self.config['learn']['tune_freq'] == 0 or epoch == self.config['learn']['num_epochs']:
                self.save_net_and_optimizer(epoch)
                self.run_sparse_tuning(epoch)

            self.scheduler.step()

        self.print_and_log('Finish learning ...')
        self.remove_log_handlers()

    def meta_train_test(self, epoch: int):
        start_time = time.time()

        # Setting network for training mode.
        self.net.train()

        # List for batch losses.
        loss_list = list()

        num_epochs = self.config['learn']['num_epochs']
        meta_loaders_len = len(self.meta_loaders)

        for i, ml in enumerate(self.meta_loaders):
            train_len = max(len(ml['train']), ml['max_iterations'] or 0)
            test_data_cycle = cycle_iterable(ml['test'])
            for j, train_data in enumerate(ml['train']):
                if j == train_len:
                    break
                loss = self.meta_train_test_step(train_data, next(test_data_cycle))
                self.print_and_log('Ep: %d/%d, data: %d/%d, it: %d/%d, loss: %.4f' %
                                   (epoch, num_epochs, i, meta_loaders_len, j, train_len, loss))
                loss_list.append(loss)

        avg_loss = np.mean(loss_list)

        # Printing epoch loss.
        self.print_and_log('Ep: %d/%d, avg loss: %.4f' % (epoch, num_epochs, avg_loss), end='\n')

        end_time = time.time()

        self.write_to_csv(
            FILENAMES['train_loss'],
            ['epoch', 'duration', 'loss'],
            {'epoch': epoch, 'duration': (end_time - start_time) * 10 ** 3, 'loss': avg_loss}
        )

    def tune_train_test(self, epoch: int, tune_loader: DatasetLoaderItem):
        start_time = time.time()

        labels, preds, names = self.tune_train_test_process(epoch, tune_loader)

        sparsity_mode = tune_loader['sparsity_mode']

        score = self.calc_and_log_metrics(labels, preds, f'"{sparsity_mode}"')

        end_time = time.time()

        row = {'epoch': epoch, 'sparsity_mode': sparsity_mode, 'duration': (end_time - start_time) * 10 ** 3}
        row.update(score)
        self.write_to_csv(
            FILENAMES['tuned_score'],
            ['epoch', 'sparsity_mode', 'duration'] + sorted(score.keys()),
            row
        )

        if epoch == self.config['learn']['num_epochs']:
            for pred, name in zip(preds, names):
                self.save_prediction(pred, name, sparsity_mode)

    def run_sparse_tuning(self, epoch: int):
        for tl in self.tune_loaders:
            if tl['sparsity_mode'] == 'point':
                sparsity_unit = 'points'
            elif tl['sparsity_mode'] == 'grid':
                sparsity_unit = 'spacing'
            elif tl['sparsity_mode'] in ['contour', 'skeleton', 'region']:
                sparsity_unit = 'density'
            else:
                sparsity_unit = 'sparsity'

            if tl['sparsity_mode'] == 'dense':
                self.print_and_log('Evaluating "%s" (%d-shot) ...' % (tl['sparsity_mode'], tl['n_shots']))
            else:
                self.print_and_log('Evaluating "%s" (%d-shot, %d-%s) ...' %
                                   (tl['sparsity_mode'], tl['n_shots'], tl['sparsity_value'], sparsity_unit))

            self.tune_train_test(epoch, tl)

            print()

    def check_output_and_ckpt_dir(self) -> bool:
        return os.path.exists(self.output_path) and os.path.exists(self.ckpt_path)

    def create_output_and_ckpt_dir(self):
        check_mkdir(self.config['save']['output_path'])
        check_mkdir(self.output_path)
        check_mkdir(self.config['save']['ckpt_path'])
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
        logging.info(start.replace("\n", "") + message + end.replace("\n", ""))
        print(start + message + end)

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
        with open(os.path.join(self.ckpt_path, FILENAMES['checkpoint']), 'r') as ckpt_file:
            self.checkpoint: dict = json.load(ckpt_file)

    def update_checkpoint(self, data: dict):
        with open(os.path.join(self.ckpt_path, FILENAMES['checkpoint']), 'w') as ckpt_file:
            self.checkpoint.update(data)
            json.dump(self.checkpoint, ckpt_file, indent=4)

    def save_config_and_net(self):
        if self.config['learn']['should_resume']:
            i = 1
            while True:
                suffix = str(i) if i != 1 else ''
                filename = f'{suffix}.'.join(FILENAMES['config'].rsplit('.', 1))
                filename = os.path.join(self.output_path, filename)
                if os.path.isfile(filename):
                    i += 1
                    continue
                with open(filename, "w") as config_file:
                    json.dump(self.config, config_file, indent=4)
                break
        else:
            with open(os.path.join(self.output_path, FILENAMES['config']), "w") as config_file:
                json.dump(self.config, config_file, indent=4)
            with open(os.path.join(self.output_path, FILENAMES['net_text']), "w") as net_file:
                n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                net_file.write('# of parameters: ' + str(n_params) + '\n\n' + str(self.net))

    def calc_and_log_metrics(self, labels: list[NDArray], preds: list[NDArray], message: str,
                             start: str = '', end: str = '') -> dict:
        score, score_text, name = self.func_calc_metrics(labels, preds)
        self.print_and_log(f'{name} - {message}: {score_text}', start, end)
        return score

    def save_all_tune_mask(self, tune_loader: DataLoader, sparsity_mode: str):
        check_mkdir(os.path.join(self.output_path, FILENAMES['tune_masks_folder']))

        # Iterating over tune batches for saving.
        for i, data in enumerate(tune_loader):

            # Obtaining images, dense labels, sparse labels and paths for batch.
            _, _, y_sparse, img_name = data

            for j in range(len(img_name)):
                stored_sparse = y_sparse[j].cpu().squeeze().numpy() + 1
                stored_sparse = (stored_sparse * (255 / stored_sparse.max())).astype(np.uint8)
                io.imsave(os.path.join(self.output_path,
                                       FILENAMES['tune_masks_folder'],
                                       f'{img_name[j]} - {sparsity_mode}.png'),
                          stored_sparse)

    def save_prediction(self, prediction: NDArray, filename: str, sparsity_mode: str):
        check_mkdir(os.path.join(self.output_path, FILENAMES['prediction_folder']))

        stored_prediction = (prediction * (255 / prediction.max())).astype(np.uint8)
        io.imsave(
            os.path.join(self.output_path, FILENAMES['prediction_folder'], f'{filename} - {sparsity_mode}.png'),
            stored_prediction)

    def save_net_and_optimizer(self, epoch: int = 0):
        prefix = f'ep{epoch}_' if epoch != 0 else ''
        torch.save(self.net.state_dict(),
                   os.path.join(self.ckpt_path, prefix + FILENAMES['net_state']))
        torch.save(self.meta_optimizer.state_dict(),
                   os.path.join(self.ckpt_path, prefix + FILENAMES['optimizer_state']))

    def load_net_and_optimizer(self, epoch: int = 0):
        prefix = f'ep{epoch}_' if epoch != 0 else ''
        self.net.load_state_dict(
            torch.load(os.path.join(self.ckpt_path, prefix + FILENAMES['net_state'])))
        self.meta_optimizer.load_state_dict(
            torch.load(os.path.join(self.ckpt_path, prefix + FILENAMES['optimizer_state'])))
