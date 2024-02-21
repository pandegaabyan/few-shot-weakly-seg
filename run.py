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
    check_git_clean,
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
        "max_items": None,
        "split_val_size": 0.2,
        "split_val_fold": val_fold,
        "split_test_size": 0.2,
        "split_test_fold": test_fold,
        "cache_data": True,
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
    learner.set_initial_messages(["Command " + " ".join(sys.argv)])
    init_ok = learner.init()
    if not init_ok:
        return None, None

    trainer = Trainer(
        max_epochs=new_config["learn"]["num_epochs"],
        check_val_every_n_epoch=new_config["learn"].get("val_freq", 1),
        callbacks=make_callbacks(
            new_config["callbacks"],
            learner.ckpt_path,
            new_config["learn"].get("val_freq", 1),
        ),
        logger=False,
        # profiler="simple"
    )

    return (learner, trainer)


def run_basic(
    config: ConfigUnion,
    dummy: bool,
    resume: bool = False,
    fit_only: bool = False,
    test_only: bool = False,
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
            project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
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
    if not fit_only:
        trainer.test(learner, ckpt_path=ckpt_path if test_only else "best")

    if use_wandb:
        wandb.finish()


def run_sweep(config: ConfigUnion, dummy: bool):
    sweep_config = {
        "method": "random",
        "count_per_agent": 3,
        "parameters": {
            "opt_lr": {
                "distribution": "log_uniform_values",
                "min": 0.00001,
                "max": 0.1,
            },
            "opt_weight_decay": {
                "distribution": "log_uniform_values",
                "min": 0.00001,
                "max": 0.1,
            },
            "opt_beta_0": {"values": [0.5, 0.9, 0.99]},
            "opt_beta_1": {"values": [0.99, 0.999, 0.9999]},
            "sch_gamma": {"distribution": "uniform", "min": 0.05, "max": 0.95},
            "dataset_fold": {"values": [0, 1, 2, 3]},
        },
    }

    ref_ckpt_path = config["learn"].get("ref_ckpt_path")
    ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

    sweep_config = initialize_sweep(config, sweep_config, dummy)
    config["wandb"]["sweep_id"] = sweep_config["sweep_id"]  # type: ignore

    def train():
        config["learn"]["run_name"] = make_run_name(config["learn"]["exp_name"])

        wandb.init(
            tags=config.get("wandb", {}).get("tags"),
            group=config["learn"]["exp_name"],
            name=config["learn"]["run_name"],
            job_type=config.get("wandb", {}).get("job_type"),
        )
        ref_config = wandb.config
        config["optimizer"]["lr"] = ref_config["opt_lr"]
        config["optimizer"]["lr_bias"] = ref_config["opt_lr"] * 2
        config["optimizer"]["weight_decay"] = ref_config["opt_weight_decay"]
        config["optimizer"]["betas"] = (
            ref_config["opt_beta_0"],
            ref_config["opt_beta_1"],
        )
        config["scheduler"]["gamma"] = ref_config["sch_gamma"]
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
        project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
        count=sweep_config.get("count_per_agent"),
    )


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["basic", "fit", "test", "sweep"]),
    default="basic",
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
        "basic",
        "fit",
        "test",
        "sweep",
    ],
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    configs: list[tuple[str, str]],
):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")

    config_mode = "fit" if mode == "basic" else mode
    config = make_config(
        mode=config_mode, dummy=dummy, use_wandb=not no_wandb, learner="simple"
    )
    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    if mode == "basic":
        run_basic(config, dummy, resume=resume)
    elif mode == "fit":
        run_basic(config, dummy, resume=resume, fit_only=True)
    elif mode == "test":
        run_basic(config, dummy, resume=resume, test_only=True)
    elif mode == "sweep":
        run_sweep(config, dummy)


if __name__ == "__main__":
    main()
