import sys
from typing import Literal

import click
from pytorch_lightning import Trainer

import wandb
from config.config_maker import make_config, make_run_name
from config.config_type import ConfigSimpleLearner, ConfigUnion
from config.constants import WANDB_SETTINGS
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.simple_unet import SimpleUnet
from learners.typings import SimpleLearnerKwargs
from runners.callbacks import make_callbacks
from runners.sweeps import initialize_sweep
from tasks.optic_disc_cup.datasets import RimOneSimpleDataset
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU
from utils.logging import (
    get_configuration,
    get_full_ckpt_path,
)
from utils.utils import parse_string
from utils.wandb import wandb_login
from wandb import util as wandb_util


def rim_one_simple_dataset(
    val_fold: int = 0, test_fold: int = 0
) -> tuple[type[SimpleDataset], SimpleDatasetKwargs]:
    rim_one_kwargs: SimpleDatasetKwargs = {
        "seed": 0,
        "max_items": 10,
        "split_val_size": 0.2,
        "split_val_fold": val_fold,
        "split_test_size": 0.2,
        "split_test_fold": test_fold,
        "dataset_name": "RIM-ONE",
    }

    return (RimOneSimpleDataset, rim_one_kwargs)


def make_learner_and_trainer(
    config: ConfigUnion,
    dummy: bool,
    resume: bool = False,
    dataset_fold: int = 0,
    learner_ckpt: str | None = None,
) -> tuple[BaseLearner | None, Trainer | None]:
    dataset_list = [rim_one_simple_dataset(dataset_fold, dataset_fold)]
    for ds in dataset_list:
        ds[1]["max_items"] = 10 if dummy else None

    new_config: ConfigSimpleLearner = config  # type: ignore
    kwargs: SimpleLearnerKwargs = {
        "config": new_config,
        "dataset_list": dataset_list,
        "loss": DiscCupLoss("ce"),
        "metric": DiscCupIoU(),
        "resume": resume,
        "force_clear_dir": True,
    }
    if learner_ckpt is None:
        learner = SimpleUnet(**kwargs)
    else:
        learner = SimpleUnet.load_from_checkpoint(learner_ckpt, **kwargs)
    learner.set_initial_messages(["Command: " + " ".join(sys.argv)])
    init_ok = learner.init()
    if not init_ok:
        return None, None

    trainer = Trainer(
        max_epochs=new_config["learn"]["num_epochs"],
        callbacks=make_callbacks(
            new_config["callbacks"],
            learner.ckpt_path,
            new_config["learn"].get("val_freq", 1),
        ),
        logger=False,
    )

    return (learner, trainer)


def run_basic(
    config: ConfigUnion, dummy: bool, resume: bool = False, test_only: bool = False
):
    exp_name = config["learn"]["exp_name"]
    run_name = config["learn"]["run_name"]

    use_wandb = config.get("wandb") is not None
    if use_wandb:
        wandb_login()
        if resume:
            prev_config = get_configuration(exp_name, run_name)
            run_id = prev_config["config"]["wandb"]["run_id"]
        else:
            run_id = wandb_util.generate_id()
        wandb.init(
            config=dict(config),
            id=run_id,
            tags=config.get("wandb", {}).get("tags"),
            project=WANDB_SETTINGS["project"],
            group=exp_name,
            name=run_name,
            job_type=config.get("wandb", {}).get("job_type"),
            resume="must" if resume else None,
        )
        config["wandb"]["run_id"] = run_id  # type: ignore

    ref_ckpt_path = config["learn"].get("ref_ckpt_path")
    if (resume and not test_only) or (test_only and ref_ckpt_path is None):
        ckpt_path = get_full_ckpt_path(exp_name, run_name, "last.ckpt")
    else:
        ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

    learner, trainer = make_learner_and_trainer(
        config, dummy, resume=resume, learner_ckpt=ckpt_path
    )
    if learner is None or trainer is None:
        return

    if not test_only:
        trainer.fit(learner, ckpt_path=ckpt_path if resume else None)
    trainer.test(learner, ckpt_path=ckpt_path if test_only else "best")

    if use_wandb:
        wandb.finish()


def run_sweep(config: ConfigUnion, dummy: bool):
    sweep_config = {
        "method": "random",
        "count": 5,
        "parameters": {
            "optimizer_lr": {"distribution": "uniform", "min": 0.0001, "max": 0.01},
            "scheduler_step_size": {"values": [50, 100, 150, 200]},
            "dataset_fold": {"values": [0, 1, 2, 3]},
        },
    }

    ref_ckpt_path = config["learn"].get("ref_ckpt_path")
    ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

    sweep_config = initialize_sweep(config, sweep_config)

    def train():
        config["learn"]["run_name"] = make_run_name()

        wandb.init(
            tags=config.get("wandb", {}).get("tags"),
            group=config["learn"]["exp_name"],
            name=config["learn"]["run_name"],
            job_type=config.get("wandb", {}).get("job_type"),
        )
        ref_config = wandb.config
        config["optimizer"]["lr"] = ref_config["optimizer_lr"]
        config["optimizer"]["lr_bias"] = ref_config["optimizer_lr"] * 2
        config["scheduler"]["step_size"] = ref_config["scheduler_step_size"]
        dataset_fold = ref_config["dataset_fold"]

        learner, trainer = make_learner_and_trainer(
            config, dummy, dataset_fold=dataset_fold, learner_ckpt=ckpt_path
        )
        if learner is None or trainer is None:
            return

        trainer.fit(learner)

        wandb.finish()

    wandb.agent(
        sweep_config["sweep_id"],
        function=train,
        project=WANDB_SETTINGS["project"],
        count=sweep_config.get("count"),
    )


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)  # Need to test this
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["train", "test", "sweep"]),
    default="train",
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
def main(
    mode: Literal[
        "train",
        "test",
        "sweep",
    ],
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    configs: list[tuple[str, str]],
):
    config = make_config(
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner="simple"
    )
    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    if mode == "train":
        run_basic(config, dummy, resume=resume)
    elif mode == "test":
        run_basic(config, dummy, resume=resume, test_only=True)
    elif mode == "sweep":
        run_sweep(config, dummy)


if __name__ == "__main__":
    main()