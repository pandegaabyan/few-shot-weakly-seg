import copy
import gc
import time
from typing import Type, Iterator, Iterable

from torch import cuda, optim, nn

from config.config_type import (
    AllConfig,
    DataConfig,
    DataTuneConfig,
    LearnConfig,
    WeaselConfig,
    ProtoSegConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    GuidedNetsConfig,
)
from data.dataset_loaders import DatasetLoaderParamReduced
from data.simple_dataset import SimpleDataset
from data.types import SimpleDatasetKeywordArgs
from learners.guidednets import GuidedNetsLearner
from learners.losses import CustomLoss
from learners.protoseg import ProtoSegLearner
from learners.simple_learner import SimpleLearner
from learners.weasel import WeaselLearner
from models.u_net import UNet
from tasks.optic_disc_cup.datasets import (
    DrishtiDataset,
    RimOneDataset,
    RimOneSimpleDataset,
    DrishtiSimpleDataset,
)
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import calc_disc_cup_iou


def get_base_config() -> AllConfig:
    data_config: DataConfig = {
        "num_classes": 3,
        "num_channels": 3,
        "num_workers": 0,
        "batch_size": 1,
        "resize_to": (256, 256),
    }
    data_tune_config: DataTuneConfig = {
        "shot_list": [10],
        "sparsity_dict": {
            "point": [10],
            "grid": [25],
            "contour": [1],
            "skeleton": [1],
            "region": [1],
            "point_old": [10],
            "grid_old": [25],
        },
    }
    learn_config: LearnConfig = {
        "should_resume": False,
        "use_gpu": True,
        "num_epochs": 200,
        "exp_name": "",
        "tune_freq": 40,
        "test_freq": 100,
    }
    loss_config: LossConfig = {"type": "ce", "ignored_index": -1}
    optimizer_config: OptimizerConfig = {
        "lr": 1e-3,
        "lr_bias": 2 * 1e-3,
        "weight_decay": 5e-5,
        "weight_decay_bias": 0,
        "betas": (0.9, 0.99),
    }
    scheduler_config: SchedulerConfig = {"step_size": 150, "gamma": 0.2}
    weasel_config: WeaselConfig = {
        "use_first_order": False,
        "update_param_step_size": 0.3,
        "tune_epochs": 40,
        "tune_test_freq": 8,
    }
    protoseg_config: ProtoSegConfig = {"embedding_size": 8}
    guidednets_config: GuidedNetsConfig = {"embedding_size": 32}
    all_config: AllConfig = {
        "data": data_config,
        "data_tune": data_tune_config,
        "learn": learn_config,
        "loss": loss_config,
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "weasel": weasel_config,
        "protoseg": protoseg_config,
        "guidednets": guidednets_config,
    }
    return all_config


def get_meta_loader_params_list() -> list[DatasetLoaderParamReduced]:
    rim_one_sparsity_params: dict = {
        "point_dot_size": 5,
        "grid_dot_size": 4,
        "contour_radius_dist": 4,
        "contour_radius_thick": 2,
        "skeleton_radius_thick": 4,
        "region_compactness": 0.5,
    }
    rim_one_meta_loader_params: DatasetLoaderParamReduced = {
        "dataset_class": RimOneDataset,
        "dataset_kwargs": {
            "split_seed": 0,
            "split_test_size": 0.2,
            "num_shots": -1,
            "sparsity_mode": "random",
            "sparsity_value": "random",
            "sparsity_params": rim_one_sparsity_params,
        },
    }
    return [rim_one_meta_loader_params]


def get_tune_loader_params() -> DatasetLoaderParamReduced:
    drishti_sparsity_params: dict = {
        "point_dot_size": 4,
        "grid_dot_size": 4,
        "contour_radius_dist": 4,
        "contour_radius_thick": 1,
        "skeleton_radius_thick": 3,
        "region_compactness": 0.5,
    }
    drishti_tune_loader_params: DatasetLoaderParamReduced = {
        "dataset_class": DrishtiDataset,
        "dataset_kwargs": {
            "split_seed": 0,
            "split_test_size": 0.2,
            "sparsity_params": drishti_sparsity_params,
        },
    }
    return drishti_tune_loader_params


def get_adam_optimizer_and_step_scheduler(
    net_named_params: Iterator | Iterable, all_config: AllConfig
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    adam_optimizer = optim.Adam(
        [
            {
                "params": [
                    param for name, param in net_named_params if name[-4:] == "bias"
                ],
                "lr": all_config["optimizer"]["lr_bias"],
                "weight_decay": all_config["optimizer"]["weight_decay_bias"],
            },
            {
                "params": [
                    param for name, param in net_named_params if name[-4:] != "bias"
                ],
                "lr": all_config["optimizer"]["lr"],
                "weight_decay": all_config["optimizer"]["weight_decay"],
            },
        ],
        betas=all_config["optimizer"]["betas"],
    )

    step_scheduler = optim.lr_scheduler.StepLR(
        adam_optimizer,
        step_size=all_config["scheduler"]["step_size"],
        gamma=all_config["scheduler"]["gamma"],
    )

    return adam_optimizer, step_scheduler


def run_clean_weasel_learning(
    all_config: AllConfig,
    meta_params: list[DatasetLoaderParamReduced],
    tune_param: DatasetLoaderParamReduced,
    calc_loss: CustomLoss | None = None,
    tune_only: bool = False,
    tune_epochs: list[int] | None = None,
):
    net = UNet(all_config["data"]["num_channels"], all_config["data"]["num_classes"])

    adam_optimizer, step_scheduler = get_adam_optimizer_and_step_scheduler(
        net.named_parameters(), all_config
    )

    learner = WeaselLearner(
        net,
        all_config,
        meta_params,
        tune_param,
        calc_disc_cup_iou,
        calc_loss=calc_loss,
        optimizer=adam_optimizer,
        scheduler=step_scheduler,
    )

    try:
        if tune_only:
            learner.retune(tune_epochs)
        else:
            learner.learn()
    except BaseException as e:
        learner.log_error()
        raise e
    finally:
        learner.remove_log_handlers()
        del net, adam_optimizer, step_scheduler, learner
        gc.collect()
        cuda.empty_cache()


def run_clean_protoseg_learning(
    all_config: AllConfig,
    meta_params: list[DatasetLoaderParamReduced],
    tune_param: DatasetLoaderParamReduced,
    calc_loss: CustomLoss | None = None,
    tune_only: bool = False,
    tune_epochs: list[int] | None = None,
):
    net = UNet(
        all_config["data"]["num_channels"], all_config["protoseg"]["embedding_size"]
    )

    adam_optimizer, step_scheduler = get_adam_optimizer_and_step_scheduler(
        net.named_parameters(), all_config
    )

    learner = ProtoSegLearner(
        net,
        all_config,
        meta_params,
        tune_param,
        calc_disc_cup_iou,
        calc_loss=calc_loss,
        optimizer=adam_optimizer,
        scheduler=step_scheduler,
    )

    try:
        if tune_only:
            learner.retune(tune_epochs)
        else:
            learner.learn()
    except BaseException as e:
        learner.log_error()
        raise e
    finally:
        learner.remove_log_handlers()
        del net, adam_optimizer, step_scheduler, learner
        gc.collect()
        cuda.empty_cache()


def run_clean_guidednets_learning(
    all_config: AllConfig,
    meta_params: list[DatasetLoaderParamReduced],
    tune_param: DatasetLoaderParamReduced,
    calc_loss: CustomLoss | None = None,
    tune_only: bool = False,
    tune_epochs: list[int] | None = None,
    use_original_way: bool = False,
):
    embedding_size = all_config["guidednets"]["embedding_size"]

    net_image = UNet(
        all_config["data"]["num_channels"], embedding_size, prototype=True
    ).cuda()

    net_mask = UNet(1, embedding_size, prototype=True).cuda()
    # net_mask = None

    net_head = nn.Sequential(
        nn.Conv2d(embedding_size * 2, embedding_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(embedding_size, all_config["data"]["num_classes"], kernel_size=1),
    ).cuda()
    nn.init.ones_(net_head[0].weight)
    nn.init.ones_(net_head[-1].weight)

    net_merge = nn.AdaptiveAvgPool2d((1, 1)).cuda()

    net = {"image": net_image, "mask": net_mask, "merge": net_merge, "head": net_head}

    if use_original_way:
        del all_config["optimizer"]["lr_bias"], all_config["optimizer"]["betas"]
        del (
            all_config["optimizer"]["weight_decay"],
            all_config["optimizer"]["weight_decay_bias"],
        )
        all_config["scheduler"]["step_size"] = 50
        all_config["scheduler"]["gamma"] = 0.5

        calc_loss = DiscCupLoss("mce")

        net_parameters = []
        for n in net.values():
            net_parameters.extend(list(n.parameters()))

        adam_optimizer = optim.Adam(net_parameters, all_config["optimizer"]["lr"])
        step_scheduler = optim.lr_scheduler.StepLR(
            adam_optimizer,
            all_config["scheduler"]["step_size"],
            gamma=all_config["scheduler"]["gamma"],
        )
    else:
        net_named_parameters = []
        for n in net.values():
            net_named_parameters.extend(list(n.named_parameters()))

        adam_optimizer, step_scheduler = get_adam_optimizer_and_step_scheduler(
            net_named_parameters, all_config
        )

    learner = GuidedNetsLearner(
        net,
        all_config,
        meta_params,
        tune_param,
        calc_disc_cup_iou,
        calc_loss=calc_loss,
        optimizer=adam_optimizer,
        scheduler=step_scheduler,
    )

    try:
        if tune_only:
            learner.retune(tune_epochs)
        else:
            learner.learn()
    except BaseException as e:
        learner.log_error()
        raise e
    finally:
        learner.remove_log_handlers()
        del (
            net_image,
            net_mask,
            net_head,
            net_merge,
            adam_optimizer,
            step_scheduler,
            learner,
        )
        gc.collect()
        cuda.empty_cache()


def run_clean_simple_learning(
    all_config: AllConfig,
    dataset_class: Type[SimpleDataset],
    dataset_kwargs: SimpleDatasetKeywordArgs,
    test_dataset_class: Type[SimpleDataset] | None = None,
    test_dataset_kwargs: SimpleDatasetKeywordArgs | None = None,
    calc_loss: CustomLoss | None = None,
    test_only: bool = False,
    test_epochs: list[int] | None = None,
):
    net = UNet(all_config["data"]["num_channels"], all_config["data"]["num_classes"])

    adam_optimizer, step_scheduler = get_adam_optimizer_and_step_scheduler(
        net.named_parameters(), all_config
    )

    learner = SimpleLearner(
        net,
        all_config,
        dataset_class,
        dataset_kwargs,
        test_dataset_class=test_dataset_class,
        test_dataset_kwargs=test_dataset_kwargs,
        calc_loss=calc_loss,
        optimizer=adam_optimizer,
        scheduler=step_scheduler,
    )

    try:
        if test_only:
            learner.retest(test_epochs)
        else:
            learner.learn()
    except BaseException as e:
        learner.log_error()
        raise e
    finally:
        learner.remove_log_handlers()
        del net, adam_optimizer, step_scheduler, learner
        gc.collect()
        cuda.empty_cache()


def main():
    all_config = get_base_config()
    all_config["data"]["num_workers"] = 3
    # all_config['learn']['should_resume'] = True
    # all_config['data']['batch_size'] = 32
    # all_config['data']['batch_size'] = 13

    all_config["learn"]["exp_name"] = "v3 RO-DR L SL"
    run_clean_simple_learning(
        all_config,
        RimOneSimpleDataset,
        {"split_seed": 0, "split_val_size": 0.2, "split_test_size": 0.2},
        test_dataset_class=DrishtiSimpleDataset,
        test_dataset_kwargs={
            "split_seed": 0,
            "split_val_size": 0.2,
            "split_test_size": 0.2,
        },
    )

    meta_loader_params_list = get_meta_loader_params_list()
    tune_loader_params = get_tune_loader_params()

    all_config["learn"]["exp_name"] = "v3 RO-DR L WS"
    run_clean_weasel_learning(all_config, meta_loader_params_list, tune_loader_params)

    all_config["learn"]["exp_name"] = "v3 RO-DR L PS"
    run_clean_protoseg_learning(
        all_config,
        meta_loader_params_list,
        tune_loader_params,
        tune_only=True,
        tune_epochs=[40, 200],
    )

    all_config["learn"]["exp_name"] = "v3 RO-DR L GN"
    run_clean_guidednets_learning(
        all_config, meta_loader_params_list, tune_loader_params
    )

    config_items = [
        {
            "sparsity_mode": "point",
            "sparsity_value_tune": 10,
            "sparsity_value_meta": (1, 15),
        },
        {
            "sparsity_mode": "grid",
            "sparsity_value_tune": 25,
            "sparsity_value_meta": (15, 50),
        },
        {
            "sparsity_mode": "contour",
            "sparsity_value_tune": 1,
            "sparsity_value_meta": (0.2, 1),
        },
        {
            "sparsity_mode": "skeleton",
            "sparsity_value_tune": 1,
            "sparsity_value_meta": (0.2, 1),
        },
        {
            "sparsity_mode": "region",
            "sparsity_value_tune": 1,
            "sparsity_value_meta": (0.2, 1),
        },
    ]

    for config_item in config_items:
        new_config = copy.deepcopy(all_config)
        new_config["learn"][
            "exp_name"
        ] = f'v3 RO-DR L WS {config_item["sparsity_mode"]}-var'
        new_config["data_tune"]["sparsity_dict"] = {
            config_item["sparsity_mode"]: [config_item["sparsity_value_tune"]]
        }

        new_meta_loader_params_list = []
        for param in meta_loader_params_list:
            new_param = copy.deepcopy(param)
            new_param["dataset_kwargs"]["sparsity_mode"] = config_item["sparsity_mode"]
            new_param["dataset_kwargs"]["sparsity_value"] = config_item[
                "sparsity_value_meta"
            ]
            new_meta_loader_params_list.append(new_param)

        new_tune_loader_params = copy.deepcopy(tune_loader_params)

        run_clean_weasel_learning(
            new_config, new_meta_loader_params_list, new_tune_loader_params
        )

        time.sleep(60)


if __name__ == "__main__":
    main()
