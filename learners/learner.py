import os
from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.utils.data import DataLoader

from data.meta_dataset import MetaDatasets
from data.tune_loader import TuneLoaderDict
from config.config_type import AllConfig
from torchmeta.modules import MetaModule


class MetaLearner(ABC):

    def __init__(self, net: MetaModule, config: AllConfig, meta_set: MetaDatasets, tune_loader: TuneLoaderDict):
        self.config = config
        self.meta_set = meta_set
        self.tune_loader = tune_loader

        if self.config['train']["use_gpu"]:
            self.net = net.cuda()
        else:
            self.net = net

        self.meta_optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * self.config['train']['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': self.config['train']['lr'], 'weight_decay': self.config['train']['weight_decay']}
        ], betas=(self.config['train']['momentum'], 0.99))

        self.scheduler = optim.lr_scheduler.StepLR(self.meta_optimizer,
                                                   self.config['train']['lr_scheduler_step_size'],
                                                   gamma=self.config['train']['lr_scheduler_gamma'],
                                                   last_epoch=-1)

    # Training function.
    @abstractmethod
    def meta_train_test(self, epoch: int):
        pass

    @abstractmethod
    def tune_train_test(self, tune_train_loader: DataLoader, tune_test_loader: DataLoader,
                        epoch: int, sparsity_mode: str):
        pass

    def learn(self):
        print(self.net)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('# of parameters: ' + str(n_params))

        # Loading optimizer state in case of resuming training.
        if self.config['train']['snapshot'] == '':
            curr_epoch = 1

        else:
            print('Training resuming from epoch ' + str(self.config['train']['snapshot']) + '...')
            self.net.load_state_dict(
                torch.load(os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'], 'meta.pth')))
            self.meta_optimizer.load_state_dict(torch.load(
                os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name'], 'opt_meta.pth')))
            curr_epoch = int(self.config['train']['snapshot']) + 1

        # Making sure checkpoint and output directories are created.
        self.check_mkdir(self.config['save']['ckpt_path'])
        self.check_mkdir(os.path.join(self.config['save']['ckpt_path'], self.config['save']['exp_name']))
        self.check_mkdir(self.config['save']['output_path'])
        self.check_mkdir(os.path.join(self.config['save']['output_path'], self.config['save']['exp_name']))

        # Iterating over epochs.
        for epoch in range(curr_epoch, self.config['train']['epoch_num'] + 1):

            # Meta training on source datasets.
            self.meta_train_test(epoch)

            if epoch % self.config['train']['test_freq'] == 0:
                self.run_sparse_tuning(epoch)

            self.scheduler.step()

    def run_sparse_tuning(self, epoch: int):

        # Tuning/testing on points.
        for dict_point in self.tune_loader['point']:
            n_shots = dict_point['n_shots']
            sparsity = dict_point['sparsity']

            print('    Evaluating \'points\' (%d-shot, %d-points)...' % (n_shots, sparsity))

            self.tune_train_test(dict_point['train'],
                                 dict_point['test'],
                                 epoch,
                                 'points_(%d-shot_%d-points)' % (n_shots, sparsity))

        # Tuning/testing on grid.
        for dict_grid in self.tune_loader['grid']:
            n_shots = dict_grid['n_shots']
            sparsity = dict_grid['sparsity']

            print('    Evaluating \'grid\' (%d-shot, %d-spacing)...' % (n_shots, sparsity))

            self.tune_train_test(dict_grid['train'],
                                 dict_grid['test'],
                                 epoch,
                                 'grid_(%d-shot_%d-spacing)' % (n_shots, sparsity))

        # Tuning/testing on contours.
        for dict_contour in self.tune_loader['contour']:
            n_shots = dict_contour['n_shots']
            sparsity = dict_contour['sparsity']

            print('    Evaluating \'contours\' (%d-shot, %.2f-density)...' % (n_shots, sparsity))

            self.tune_train_test(dict_contour['train'],
                                 dict_contour['test'],
                                 epoch,
                                 'contours_(%d-shot_%.2f-density)' % (n_shots, sparsity))

        # Tuning/testing on skels.
        for dict_skeleton in self.tune_loader['skeleton']:
            n_shots = dict_skeleton['n_shots']
            sparsity = dict_skeleton['sparsity']

            print('    Evaluating \'skels\' (%d-shot, %.2f-skels)...' % (n_shots, sparsity))

            self.tune_train_test(dict_skeleton['train'],
                                 dict_skeleton['test'],
                                 epoch,
                                 'skels_(%d-shot_%.2f-skels)' % (n_shots, sparsity))

        # Tuning/testing on regions.
        for dict_region in self.tune_loader['region']:
            n_shots = dict_region['n_shots']
            sparsity = dict_region['sparsity']

            print('    Evaluating \'regions\' (%d-shot, %.2f-regions)...' % (n_shots, sparsity))

            self.tune_train_test(dict_region['train'],
                                 dict_region['test'],
                                 epoch,
                                 'regions_(%d-shot_%.2f-regions)' % (n_shots, sparsity))

        # Tuning/testing on dense.
        for dict_dense in self.tune_loader['dense']:
            n_shots = dict_dense['n_shots']

            print('    Evaluating \'dense\' (%d-shot)...' % n_shots)

            self.tune_train_test(dict_dense['train'],
                                 dict_dense['test'],
                                 epoch,
                                 'dense_(%d-shot)' % n_shots)

    @staticmethod
    def check_mkdir(dir_name: str):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    def prepare_meta_batch(self, index: int):

        # Acquiring training and test data.
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

            if self.config['train']['use_gpu']:
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
