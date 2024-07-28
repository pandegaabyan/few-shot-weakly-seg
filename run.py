import gc

import click
from torch import cuda

from config.config_maker import make_config
from config.config_type import AllConfig
from data.get_meta_datasets import MetaDatasets, get_meta_datasets
from data.get_tune_loaders import TuneLoaderDict, get_tune_loaders
from learners.protoseg import ProtoSegLearner
from learners.weasel import WeaselLearner
from models.u_net import UNet
from tasks.optic_disc_cup.datasets import DrishtiDataset, RimOneDataset
from tasks.optic_disc_cup.metrics import calc_disc_cup_iou


def prepare_data(config: AllConfig) -> tuple[MetaDatasets, TuneLoaderDict]:
    rim_one_sparsity_params: dict = {
        "contour_radius_dist": 4,
        "contour_radius_thick": 2,
        "skeleton_radius_thick": 4,
        "region_compactness": 0.5,
    }
    meta_set = get_meta_datasets(
        [
            {
                "dataset_class": RimOneDataset,
                "num_classes": config["data"]["num_classes"],
                "resize_to": config["data"]["resize_to"],
                "kwargs": {
                    "split_seed": 0,
                    "split_test_size": 0.8,
                    "sparsity_mode": "random",
                    "sparsity_value": "random",
                    "sparsity_params": rim_one_sparsity_params,
                },
            }
        ]
    )

    drishti_sparsity_params: dict = {
        "contour_radius_dist": 4,
        "contour_radius_thick": 1,
        "skeleton_radius_thick": 3,
        "region_compactness": 0.5,
    }
    tune_loader = get_tune_loaders(
        dataset_class=DrishtiDataset,
        dataset_kwargs={
            "split_seed": 0,
            "split_test_size": 0.8,
            "sparsity_mode": "random",
            "sparsity_value": "random",
            "sparsity_params": drishti_sparsity_params,
        },
        num_classes=config["data"]["num_classes"],
        resize_to=config["data"]["resize_to"],
        shots=config["data_tune"]["list_shots"],
        point=config["data_tune"]["list_sparsity_point"],
        grid=config["data_tune"]["list_sparsity_grid"],
        contour=config["data_tune"]["list_sparsity_contour"],
        skeleton=config["data_tune"]["list_sparsity_skeleton"],
        region=config["data_tune"]["list_sparsity_region"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    return meta_set, tune_loader


@click.command()
@click.option(
    "--learner",
    "-l",
    type=click.Choice(["weasel", "protoseg", ""]),
    default="",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fit", "study"]),
    default="fit",
)
@click.option("--dummy", "-d", is_flag=True)
def main(learner, mode, dummy):
    config = make_config(learner=learner or None, mode=mode, dummy=dummy)

    meta_set, tune_loader = prepare_data(config)

    if learner == "weasel":
        net = UNet(config["data"]["num_channels"], config["data"]["num_classes"])
        learner = WeaselLearner(net, config, meta_set, tune_loader, calc_disc_cup_iou)
    elif learner == "protoseg":
        net = UNet(config["data"]["num_channels"], config["protoseg"]["embedding_size"])
        learner = ProtoSegLearner(net, config, meta_set, tune_loader, calc_disc_cup_iou)
    else:
        raise ValueError("Invalid learner")

    try:
        learner.learn()
    finally:
        net = None
        learner = None
        gc.collect()
        cuda.empty_cache()


if __name__ == "__main__":
    main()
