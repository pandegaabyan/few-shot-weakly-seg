import os
import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from skimage import io
from torch import nn
from torch.utils.data import DataLoader

from config.config_type import AllConfig
from config.constants import FILENAMES
from data.dataset_loaders import DatasetLoaderItem, DatasetLoaderParamReduced, get_meta_loaders, get_tune_loaders
from data.types import TensorDataItem
from learners.base_learner import BaseLearner
from learners.losses import CustomLoss
from learners.types import CalcMetrics, NeuralNetworks, Optimizer, Scheduler
from learners.utils import check_mkdir, cycle_iterable, get_gpu_memory, dump_json, add_suffix_to_filename


class MetaLearner(BaseLearner, ABC):

    def __init__(self,
                 net: NeuralNetworks,
                 config: AllConfig,
                 meta_params: list[DatasetLoaderParamReduced],
                 tune_param: DatasetLoaderParamReduced,
                 calc_metrics: CalcMetrics | None = None,
                 calc_loss: CustomLoss | None = None,
                 optimizer: Optimizer | None = None,
                 scheduler: Scheduler | None = None):
        super().__init__(net, config, calc_metrics, calc_loss, optimizer, scheduler)

        self.meta_params = meta_params
        self.tune_param = tune_param

        self.meta_loaders = get_meta_loaders(self.meta_params, config['data'],
                                             pin_memory=config['learn']['use_gpu'])
        self.tune_loaders = get_tune_loaders(self.tune_param, config['data'],
                                             config['data_tune'], pin_memory=config['learn']['use_gpu'])

    @abstractmethod
    def meta_train_test_step(self, train_data: TensorDataItem, test_data: TensorDataItem) -> float:
        pass

    @abstractmethod
    def tune_train_test_process(self, epoch: int,
                                tune_loader: DatasetLoaderItem) -> tuple[list[NDArray], list[NDArray], list[str]]:
        pass

    def pre_meta_train_test(self, epoch: int):
        pass

    @staticmethod
    def set_used_config() -> list[str]:
        return ['data', 'data_tune', 'learn', 'loss', 'optimizer', 'scheduler']

    def learn_process(self, epoch: int):
        self.pre_meta_train_test(epoch)
        self.meta_train_test(epoch)
        self.save_net_and_optimizer()
        self.update_checkpoint({'epoch': epoch})

        tune_freq = self.config['learn'].get('tune_freq')
        if tune_freq is not None and epoch % tune_freq == 0 or epoch == self.config['learn']['num_epochs']:
            self.save_net_and_optimizer(epoch)
            self.run_sparse_tuning(epoch)

        self.scheduler.step()

    def retune(self, epochs: list[int] | None = None):

        gpu_percent, gpu_total = self.initialize_gpu_usage()

        ok = self.check_output_and_ckpt_dir()
        if not ok:
            print('No data from previous learning')
            return

        if epochs is None:
            num_epochs = self.config['learn']['num_epochs']
            tune_freq = self.config['learn'].get('tune_freq')
            if tune_freq is None:
                print('No epochs argument and no tune_freq in config')
                return
            else:
                epochs = list(range(tune_freq, num_epochs + 1, tune_freq))

        self.initialize_log()
        self.save_configuration(False)

        self.print_and_log(f'Start retuning on epochs: {epochs} ...', end='\n')
        if self.config['learn']["use_gpu"]:
            self.print_and_log('Using GPU with total memory %dMiB, %.2f%% is already used' %
                               (gpu_total, gpu_percent))

        for epoch in epochs:
            self.print_and_log('Ep: %d' % epoch)
            try:
                self.load_net_and_optimizer(epoch)
            except FileNotFoundError:
                self.print_and_log('Checkpoint not found, continue ...')
                continue
            self.run_sparse_tuning(epoch)

        self.print_and_log('Finish retuning ...')
        self.remove_log_handlers()

    def meta_train_test(self, epoch: int):
        start_time = time.time()

        # Setting network for training mode.
        if isinstance(self.net, nn.Module):
            self.net.train()
        else:
            for net in self.net.values():
                net.train()

        # List for batch losses.
        loss_list = list()

        num_epochs = self.config['learn']['num_epochs']
        meta_loaders_len = len(self.meta_loaders)

        for i, ml in enumerate(self.meta_loaders):
            train_len = len(ml['train'])
            test_data_cycle = cycle_iterable(ml['test'])
            for j, train_data in enumerate(ml['train']):
                if j == train_len:
                    break
                loss = self.meta_train_test_step(train_data, next(test_data_cycle))
                self.print_and_log('Ep: %d/%d, data: %d/%d, it: %d/%d, loss: %.4f' %
                                   (epoch, num_epochs, i + 1, meta_loaders_len, j + 1, train_len, loss))
                loss_list.append(loss)

        gpu_percent, _ = get_gpu_memory()

        total_loss = np.sum(loss_list)

        # Printing epoch loss.
        self.print_and_log('Ep: %d/%d, total loss: %.4f' % (epoch, num_epochs, total_loss), end='\n')

        end_time = time.time()

        self.write_to_csv(
            FILENAMES['train_loss'],
            ['epoch', 'duration', 'post_gpu_percent', 'loss'],
            {'epoch': epoch, 'duration': (end_time - start_time) * 10 ** 3,
             'post_gpu_percent': gpu_percent - self.initial_gpu_percent, 'loss': total_loss}
        )

    def tune_train_test(self, epoch: int, tune_loader: DatasetLoaderItem):
        start_time = time.time()

        labels, preds, names = self.tune_train_test_process(epoch, tune_loader)

        gpu_percent, _ = get_gpu_memory()

        tune_param_str = self.encode_tune_param(tune_loader)
        score = self.calc_and_log_metrics(labels, preds, tune_param_str)

        end_time = time.time()

        row = {
            'epoch': epoch,
            'n_shots': tune_loader['n_shots'],
            'sparsity_mode': tune_loader['sparsity_mode'],
            'sparsity_value': tune_loader['sparsity_value'],
            'duration': (end_time - start_time) * 10 ** 3,
            'post_gpu_percent': gpu_percent - self.initial_gpu_percent
        }
        row.update(score)
        self.write_to_csv(
            FILENAMES['tuned_score'],
            ['epoch', 'n_shots', 'sparsity_mode', 'sparsity_value', 'duration', 'post_gpu_percent']
            + sorted(score.keys()),
            row
        )

        if epoch == self.config['learn']['num_epochs']:
            for pred, name in zip(preds, names):
                self.save_prediction(pred, name, tune_param_str)

    def run_sparse_tuning(self, epoch: int):
        for tl in self.tune_loaders:
            tune_param_str = self.encode_tune_param(tl)
            self.print_and_log(f'Evaluating "{tune_param_str}" ...')
            self.tune_train_test(epoch, tl)

            print()

    @staticmethod
    def encode_tune_param(tune_loader: DatasetLoaderItem) -> str:
        n_shots = tune_loader['n_shots']
        sparsity_mode = tune_loader['sparsity_mode']
        sparsity_value = tune_loader['sparsity_value']
        if type(sparsity_value) is tuple:
            sparsity_value = f'{sparsity_value[0]},{sparsity_value[1]}'
        if tune_loader['sparsity_mode'] == 'dense':
            return f'shot={n_shots} dense'
        else:
            return f'shot={n_shots} {sparsity_mode}={sparsity_value}'

    @staticmethod
    def dictify_loader_param(param: DatasetLoaderParamReduced) -> dict:
        new_param: dict = param.copy()
        new_param['dataset_class'] = str(param['dataset_class']).replace('\'', '')
        return new_param

    def get_dataset_params_dict(self) -> dict:
        return {
            'meta': [self.dictify_loader_param(mp) for mp in self.meta_params],
            'tune': self.dictify_loader_param(self.tune_param)
        }

    def save_configuration(self, is_new: bool):
        dataset_params = self.get_dataset_params_dict()
        optimization_data = self.get_optimization_data_dict()
        config_filepath = os.path.join(self.output_path, FILENAMES['config'])
        dataset_path = os.path.join(self.output_path, FILENAMES['dataset_config'])
        opt_path = os.path.join(self.output_path, FILENAMES['optimization_data'])
        if not is_new:
            i = 1
            while os.path.isfile(config_filepath) or os.path.isfile(dataset_path) or os.path.isfile(opt_path):
                i += 1
                config_filepath = add_suffix_to_filename(config_filepath, str(i))
                dataset_path = add_suffix_to_filename(dataset_path, str(i))
                opt_path = add_suffix_to_filename(opt_path, str(i))

        dump_json(config_filepath, self.config)
        dump_json(dataset_path, dataset_params)
        dump_json(opt_path, optimization_data)
        if is_new:
            self.save_net_as_text()

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

    def save_prediction(self, prediction: NDArray, filename: str, tune_param: str):
        check_mkdir(os.path.join(self.output_path, FILENAMES['prediction_folder']))

        stored_prediction = (prediction * (255 / prediction.max())).astype(np.uint8)
        io.imsave(
            os.path.join(self.output_path, FILENAMES['prediction_folder'], f'{filename} - {tune_param}.png'),
            stored_prediction)
