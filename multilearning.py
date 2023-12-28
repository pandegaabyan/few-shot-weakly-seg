import copy
import gc
import time
from typing import Type

from torch import cuda

from config.config_type import AllConfig, DataConfig, DataTuneConfig, LearnConfig, WeaselConfig, ProtoSegConfig
from data.dataset_loaders import DatasetLoaderParamSimple
from learners.learner import MetaLearner
from learners.protoseg import ProtoSegLearner
from learners.weasel import WeaselLearner
from models.u_net import UNet
from tasks.optic_disc_cup.datasets import DrishtiDataset, RimOneDataset
from tasks.optic_disc_cup.metrics import calc_disc_cup_iou
from torchmeta.modules import MetaModule


def get_base_config() -> AllConfig:
    data_config: DataConfig = {
        'num_classes': 3,
        'num_channels': 3,
        'num_workers': 0,
        'batch_size': 1,
        'resize_to': (256, 256)
    }
    data_tune_config: DataTuneConfig = {
        'shot_list': [10],
        'sparsity_dict': {
            'point': [10],
            'grid': [25],
            'contour': [1],
            'skeleton': [1],
            'region': [1],
            'point_old': [10],
            'grid_old': [25]
        }
    }
    learn_config: LearnConfig = {
        'should_resume': False,
        'use_gpu': True,
        'num_epochs': 200,
        'optimizer_lr': 1e-3,
        'optimizer_weight_decay': 5e-5,
        'optimizer_momentum': 0.9,
        'scheduler_step_size': 150,
        'scheduler_gamma': 0.2,
        'tune_freq': 40,
        'exp_name': ''
    }
    weasel_config: WeaselConfig = {
        'use_first_order': False,
        'update_param_step_size': 0.3,
        'tune_epochs': 40,
        'tune_test_freq': 8
    }
    protoseg_config: ProtoSegConfig = {
        'embedding_size': 4
    }
    all_config: AllConfig = {
        'data': data_config,
        'data_tune': data_tune_config,
        'learn': learn_config,
        'weasel': weasel_config,
        'protoseg': protoseg_config
    }
    return all_config


def get_meta_loader_params_list() -> list[DatasetLoaderParamSimple]:
    rim_one_sparsity_params: dict = {
        'point_dot_size': 5,
        'grid_dot_size': 4,
        'contour_radius_dist': 4,
        'contour_radius_thick': 2,
        'skeleton_radius_thick': 4,
        'region_compactness': 0.5
    }
    rim_one_meta_loader_params: DatasetLoaderParamSimple = {
        'dataset_class': RimOneDataset,
        'dataset_kwargs': {
            'split_seed': 0,
            'split_test_size': 0.2,
            'num_shots': -1,
            'sparsity_mode': 'random',
            'sparsity_value': 'random',
            'sparsity_params': rim_one_sparsity_params
        }
    }
    return [rim_one_meta_loader_params]


def get_tune_loader_params() -> DatasetLoaderParamSimple:
    drishti_sparsity_params: dict = {
        'point_dot_size': 4,
        'grid_dot_size': 4,
        'contour_radius_dist': 4,
        'contour_radius_thick': 1,
        'skeleton_radius_thick': 3,
        'region_compactness': 0.5
    }
    drishti_tune_loader_params: DatasetLoaderParamSimple = {
        'dataset_class': DrishtiDataset,
        'dataset_kwargs': {
            'split_seed': 0,
            'split_test_size': 0.2,
            'sparsity_params': drishti_sparsity_params
        }
    }
    return drishti_tune_loader_params


def run_clean_learning(learner_class: Type[MetaLearner],
                       net: MetaModule,
                       all_config: AllConfig,
                       meta_params: list[DatasetLoaderParamSimple],
                       tune_param: DatasetLoaderParamSimple,
                       tune_only: bool = False):

    learner = learner_class(net,
                            all_config,
                            meta_params,
                            tune_param,
                            calc_disc_cup_iou)

    try:
        if tune_only:
            learner.retune()
        else:
            learner.learn()
    except BaseException as e:
        learner.log_error()
        raise e
    finally:
        learner.remove_log_handlers()
        del net
        del learner
        gc.collect()
        cuda.empty_cache()


def main():
    all_config = get_base_config()
    all_config['data']['num_workers'] = 3
    # all_config['learn']['should_resume'] = True

    meta_loader_params_list = get_meta_loader_params_list()
    tune_loader_params = get_tune_loader_params()

    all_config['learn']['exp_name'] = 'WS RO-DR long v3'
    all_config['data']['batch_size'] = 14

    net = UNet(all_config['data']['num_channels'], all_config['data']['num_classes'])

    run_clean_learning(WeaselLearner, net, all_config,
                       meta_loader_params_list, tune_loader_params)

    all_config['learn']['exp_name'] = 'PS RO-DR long v3'
    all_config['data']['batch_size'] = 36

    net = UNet(all_config['data']['num_channels'], all_config['protoseg']['embedding_size'])

    run_clean_learning(ProtoSegLearner, net, all_config,
                       meta_loader_params_list, tune_loader_params, True)

    config_items = [
        {
            'sparsity_mode': 'point',
            'sparsity_value': 10
        },
        {
            'sparsity_mode': 'grid',
            'sparsity_value': 25
        },
        {
            'sparsity_mode': 'contour',
            'sparsity_value': 1
        },
        {
            'sparsity_mode': 'skeleton',
            'sparsity_value': 1
        },
        {
            'sparsity_mode': 'region',
            'sparsity_value': 1
        },
    ]

    for config_item in config_items:
        new_config = copy.deepcopy(all_config)
        new_config['learn']['exp_name'] = f'WS RO-DR long v3 {config_item["sparsity_mode"]}'
        new_config['data_tune']['sparsity_dict'] = {
            config_item['sparsity_mode']: [config_item['sparsity_value']]
        }

        new_meta_loader_params_list = []
        for param in meta_loader_params_list:
            new_param = copy.deepcopy(param)
            new_param['dataset_kwargs']['sparsity_mode'] = config_item['sparsity_mode']
            new_param['dataset_kwargs']['sparsity_value'] = config_item['sparsity_value']
            new_meta_loader_params_list.append(new_param)

        new_tune_loader_params = copy.deepcopy(tune_loader_params)

        net = UNet(all_config['data']['num_channels'], all_config['data']['num_classes'])

        run_clean_learning(WeaselLearner, net, new_config,
                           new_meta_loader_params_list, new_tune_loader_params)

        time.sleep(60)


if __name__ == '__main__':
    main()
