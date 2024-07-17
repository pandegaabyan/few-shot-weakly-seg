import sys

import click
import optuna

from config.config_maker import make_config
from config.config_type import ConfigSimpleLearner, ConfigUnion, RunMode
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.simple_unet import SimpleUnet
from learners.typings import SimpleLearnerKwargs
from runners.runner import Runner
from tasks.optic_disc_cup.datasets import RimOneSimpleDataset
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU
from utils.logging import (
    check_git_clean,
)
from utils.utils import parse_string


class MyRunner(Runner):
    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        ckpt_path: str | None = None,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[BaseLearner, dict]:
        dataset_list = self.make_dataset_list(dataset_fold)
        for ds in dataset_list:
            ds[1]["max_items"] = 6 if dummy else None

        typed_config: ConfigSimpleLearner = config  # type: ignore
        kwargs: SimpleLearnerKwargs = {
            "config": typed_config,
            "dataset_list": dataset_list,
            "loss": (DiscCupLoss, {"mode": "ce"}),
            "metric": (DiscCupIoU, {}),
            "optuna_trial": optuna_trial,
        }
        if ckpt_path is None:
            learner = SimpleUnet(**kwargs)
        else:
            learner = SimpleUnet.load_from_checkpoint(ckpt_path, **kwargs)

        learner.set_initial_messages(["Command " + " ".join(sys.argv)])

        return learner, {}

    def make_dataset_list(
        self,
        val_fold: int = 0,
    ) -> list[tuple[type[SimpleDataset], SimpleDatasetKwargs]]:
        rim_one_kwargs: SimpleDatasetKwargs = {
            "seed": 0,
            "max_items": None,
            "split_val_size": 0.2,
            "split_val_fold": val_fold,
            "split_test_size": 0.2,
            "split_test_fold": 0,
            "cache_data": True,
            "dataset_name": "RIM-ONE",
        }

        return [(RimOneSimpleDataset, rim_one_kwargs)]


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fit-test", "fit", "test", "study"]),
    default="fit-test",
)
@click.option(
    "--configs",
    "-c",
    nargs=2,
    multiple=True,
    type=(str, str),
    default=[],
    help="(key, value) for overriding config, use '/' for nesting keys",
)
@click.option(
    "--optuna_configs",
    "-oc",
    nargs=2,
    multiple=True,
    type=(str, str),
    default=[],
    help="(key, value) for overriding optuna config",
)
def main(
    mode: RunMode,
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    configs: list[tuple[str, str]],
    optuna_configs: list[tuple[str, str]],
):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")

    config = make_config(
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner="simple"
    )
    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    my_runner = MyRunner(config, dummy, resume)

    if mode in ["fit-test", "fit", "test"]:
        my_runner.run_fit_test(mode == "fit", mode == "test")

    for key, value in optuna_configs:
        my_runner.optuna_config[key] = parse_string(value)

    if mode in ["study"]:
        my_runner.run_study()


if __name__ == "__main__":
    main()
