import os
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import io
from torch import optim, Tensor
from torch.utils.data import DataLoader

from config.config_type import AllConfig
from data.few_sparse_dataset import SparsityModesNoRandom
from data.get_meta_datasets import MetaDatasets
from data.get_tune_loaders import TuneLoaderDict
from torchmeta.modules import MetaModule


class MetaLearner(ABC):

    def __init__(self,
                 net: MetaModule,
                 config: AllConfig,
                 meta_set: MetaDatasets,
                 tune_loader: TuneLoaderDict,
                 func_calc_print_metrics: Callable[[list[NDArray], list[NDArray], str], None]):
        self.config = config
        self.meta_set = meta_set
        self.tune_loader = tune_loader
        self.func_calc_print_metrics = func_calc_print_metrics

        self.output_path = os.path.join(self.config['save']['output_path'], self.config['save']['exp_name'])
        self.ckpt_path = os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'])

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
                                                   gamma=self.config['learn']['scheduler_gamma'],
                                                   last_epoch=-1)

    @abstractmethod
    def meta_train_test_step(self, dataset_indices: list[int]) -> list[float]:
        pass

    @abstractmethod
    def tune_train_test(self, tune_train_loader: DataLoader, tune_test_loader: DataLoader,
                        epoch: int, sparsity_mode: str):
        pass

    def learn(self):
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('# of parameters: ' + str(n_params))
        print()

        # TODO: store all config in json and model in string

        # Loading optimizer state in case of resuming training.
        if self.config['learn']['last_stored_epoch'] == -1:
            curr_epoch = 1

        else:
            print('Training resuming from epoch ' + str(self.config['learn']['last_stored_epoch']) + '...')
            print()
            self.load_net_and_optimizer()
            curr_epoch = self.config['learn']['last_stored_epoch'] + 1

        # Iterating over epochs.
        for epoch in range(curr_epoch, self.config['learn']['num_epochs'] + 1):

            # Meta training on source datasets.
            self.meta_train_test(epoch)

            if epoch % self.config['learn']['tune_freq'] == 0 or epoch == self.config['learn']['num_epochs']:
                self.run_sparse_tuning(epoch)

            self.scheduler.step()

    def meta_train_test(self, epoch: int):

        # Setting network for training mode.
        self.net.train()

        # List for batch losses.
        loss_list = list()

        # Iterating over batches.
        for i in range(1, self.config['learn']['meta_iterations'] + 1):
            # Randomly selecting datasets.
            perm = np.random.permutation(len(self.meta_set['train']))
            indices = perm[:self.config['learn']['meta_used_datasets']]
            print('Ep: ' + str(epoch) + ', it: ' + str(i) + ', datasets subset: ' + str(indices))

            loss = self.meta_train_test_step(list(indices))
            loss_list.extend(loss)

        # Saving meta-model.
        self.save_net_and_optimizer()
        # TODO: store timestamp, epoch, loss

        # Printing epoch loss.
        print('[epoch %d], [train loss %.4f]' % (epoch, np.asarray(loss_list).mean()))
        print()

    def run_sparse_tuning(self, epoch: int):
        sparsity_modes: list[SparsityModesNoRandom] = ['point', 'grid', 'contour', 'skeleton', 'region', 'dense']

        for sparsity_mode in sparsity_modes:
            for tl in self.tune_loader[sparsity_mode]:
                if sparsity_mode == 'dense':
                    sparsity_unit = ''
                elif sparsity_mode == 'point':
                    sparsity_unit = 'points'
                elif sparsity_mode == 'grid':
                    sparsity_unit = 'spacing'
                else:
                    sparsity_unit = 'density'

                if sparsity_mode == 'dense':
                    print('Evaluating "%s" (%d-shot) ...' % (sparsity_mode, tl['n_shots']))
                else:
                    print('Evaluating "%s" (%d-shot, %d-%s) ...' %
                          (sparsity_mode, tl['n_shots'], tl['sparsity'], sparsity_unit))

                self.tune_train_test(tl['train'], tl['test'],
                                     epoch, sparsity_mode)

                print()

    def prepare_meta_batch(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        perm_train = torch.randperm(len(self.meta_set['train'][index])).tolist()
        perm_test = torch.randperm(len(self.meta_set['test'][index])).tolist()

        for b in range(self.config['data']['batch_size']):

            d_tr = self.meta_set['train'][index][perm_train[b]]
            d_ts = self.meta_set['test'][index][perm_test[b]]

            x_tr = d_tr[0]
            y_tr = d_tr[2]
            x_ts = d_ts[0]
            y_ts = d_ts[1]

            if self.config['learn']['use_gpu']:
                x_tr = x_tr.cuda()
                y_tr = y_tr.cuda()
                x_ts = x_ts.cuda()
                y_ts = y_ts.cuda()

            x_train.append(x_tr)
            y_train.append(y_tr)
            x_test.append(x_ts)
            y_test.append(y_ts)

        x_train = torch.stack(x_train, dim=0)
        y_train = torch.stack(y_train, dim=0)
        x_test = torch.stack(x_test, dim=0)
        y_test = torch.stack(y_test, dim=0)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def check_mkdir(dir_name: str):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    def create_output_dir(self, dir_name: str):
        self.check_mkdir(self.config['save']['output_path'])
        self.check_mkdir(self.output_path)
        self.check_mkdir(os.path.join(self.output_path, dir_name))

    def save_all_tune_mask(self, tune_loader: DataLoader, sparsity_mode: str):
        dir_name = 'all_tune_masks'

        self.create_output_dir(dir_name)

        # Iterating over tune batches for saving.
        for i, data in enumerate(tune_loader):

            # Obtaining images, dense labels, sparse labels and paths for batch.
            _, _, y_sparse, img_name = data

            for j in range(len(img_name)):
                stored_sparse = y_sparse[j].cpu().squeeze().numpy() + 1
                stored_sparse = (stored_sparse * (255 / stored_sparse.max())).astype(np.uint8)
                io.imsave(os.path.join(self.output_path, dir_name, f'{img_name[j]} - {sparsity_mode}.png'),
                          stored_sparse)

    def save_prediction(self, prediction: NDArray, filename: str, sparsity_mode: str):
        dir_name = 'predictions'

        self.create_output_dir(dir_name)

        stored_prediction = (prediction * (255 / prediction.max())).astype(np.uint8)
        io.imsave(
            os.path.join(self.output_path, dir_name, f'{filename} - {sparsity_mode}.png'),
            stored_prediction)

    def save_net_and_optimizer(self):
        # Making sure checkpoint and output directories are created.
        self.check_mkdir(self.config['save']['ckpt_path'])
        self.check_mkdir(os.path.join(self.ckpt_path))

        torch.save(self.net.state_dict(), os.path.join(self.ckpt_path, 'net.pth'))
        torch.save(self.meta_optimizer.state_dict(), os.path.join(self.ckpt_path, 'meta_optimizer.pth'))

    def load_net_and_optimizer(self):
        self.net.load_state_dict(
            torch.load(os.path.join(self.ckpt_path, 'net.pth')))
        self.meta_optimizer.load_state_dict(
            torch.load(os.path.join(self.ckpt_path, 'meta_optimizer.pth')))

    def calc_print_metrics(self, labels: list[NDArray], preds: list[NDArray], message: str):
        # TODO: need update, store metrics and message
        self.func_calc_print_metrics(labels, preds, message)
