import sys
from copy import deepcopy

import click
from pytorch_lightning import Trainer

import wandb
from config.config_maker import make_config, make_run_name
from config.config_type import ConfigSimpleLearner, ConfigUnion, RunMode
from config.constants import WANDB_SETTINGS
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.simple_unet import SimpleUnet
from learners.typings import SimpleLearnerKwargs
from runners.callbacks import make_callbacks
from runners.sweeps import SweepConfigBase, initialize_sweep
from tasks.optic_disc_cup.datasets import RimOneSimpleDataset
from tasks.optic_disc_cup.losses import DiscCupLoss
from tasks.optic_disc_cup.metrics import DiscCupIoU
from utils.logging import (
    check_git_clean,
    get_configuration,
    get_full_ckpt_path,
)
from utils.utils import mean, parse_string
from utils.wandb import reset_wandb_env, wandb_login
from wandb import util as wandb_util
from wandb.sdk import wandb_setup


def rim_one_simple_dataset(
    val_fold: int = 0,
) -> tuple[type[SimpleDataset], SimpleDatasetKwargs]:
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

    return (RimOneSimpleDataset, rim_one_kwargs)


def make_learner_and_trainer(
    config: ConfigUnion,
    dummy: bool,
    resume: bool = False,
    dataset_fold: int = 0,
    learner_ckpt: str | None = None,
) -> tuple[BaseLearner | None, Trainer | None]:
    dataset_list = [rim_one_simple_dataset(dataset_fold)]

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


def run_fit_test(
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
        assert "wandb" in config
        wandb_login()
        if resume:
            prev_config = get_configuration(exp_name, run_name)
            run_id = prev_config["config"]["wandb"]["run_id"]
        else:
            run_id = wandb_util.generate_id()
        wandb.init(
            config=dict(config),
            id=run_id,
            tags=config["wandb"]["tags"],
            project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
            group=exp_name,
            name=run_name,
            job_type=config["wandb"]["job_type"],
            resume="must" if resume else None,
        )
        config["wandb"]["run_id"] = run_id

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


def run_sweep(config: ConfigUnion, dummy: bool, use_cv: bool = False, count: int = 3):
    assert "wandb" in config

    sweep_config: SweepConfigBase = {
        "method": "bayes",
        "metric": {"name": "summary/val_score", "goal": "maximize"},
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
            # "dataset_fold": {"values": [0, 1, 2, 3]},
        },
    }

    def update_config_from_ref(config: ConfigUnion, ref_config: dict):
        config["optimizer"]["lr"] = ref_config["opt_lr"]
        config["optimizer"]["lr_bias"] = ref_config["opt_lr"] * 2
        config["optimizer"]["weight_decay"] = ref_config["opt_weight_decay"]
        config["optimizer"]["betas"] = (
            ref_config["opt_beta_0"],
            ref_config["opt_beta_1"],
        )
        config["scheduler"]["gamma"] = ref_config["sch_gamma"]

    ref_ckpt_path = config["learn"].get("ref_ckpt_path")
    ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

    sweep_config = initialize_sweep(config, sweep_config, dummy, use_cv)
    config["wandb"]["sweep_id"] = sweep_config["sweep_id"]

    def train(
        config: ConfigUnion = deepcopy(config), ref_config: dict | None = None
    ) -> float | None:
        assert "wandb" in config

        if use_cv:
            sweep_parent = config["wandb"].get("sweep_parent")
            assert sweep_parent and ref_config
            dataset_fold = ref_config["dataset_fold"]
            run_name = f"{sweep_parent} F{dataset_fold}"
            project = WANDB_SETTINGS["dummy_project" if dummy else "project"]
            update_config_from_ref(config, ref_config)
        else:
            run_name = make_run_name(config["learn"]["exp_name"])
            project = None
        config["learn"]["run_name"] = run_name

        wandb.init(
            config=dict(config) if use_cv else None,
            tags=config["wandb"]["tags"],
            project=project,
            group=config["learn"]["exp_name"],
            name=run_name,
            job_type=config["wandb"]["job_type"],
            reinit=True if use_cv else None,
        )
        assert wandb.run
        config["wandb"]["run_id"] = wandb.run.id

        if not use_cv:
            ref_config = dict(wandb.config)
            dataset_fold = ref_config.get("dataset_fold", 0)
            update_config_from_ref(config, ref_config)

        learner, trainer = make_learner_and_trainer(
            config, dummy, dataset_fold=dataset_fold, learner_ckpt=ckpt_path
        )
        if learner is None or trainer is None:
            return None

        trainer.fit(learner)
        monitor = config["callbacks"].get("monitor")
        final_score = trainer.callback_metrics[monitor].item() if monitor else None

        wandb.finish()
        return final_score

    def train_cv(config: ConfigUnion = deepcopy(config)):
        num_folds = 4

        assert "wandb" in config

        config["learn"]["run_name"] = make_run_name(config["learn"]["exp_name"])

        wandb.init(
            tags=config["wandb"]["tags"] + ["sweep-parent"],
            group=config["learn"]["exp_name"],
            name=config["learn"]["run_name"],
            job_type=config["wandb"]["job_type"],
        )
        sweep_run_id = wandb.run and wandb.run.id
        ref_config = dict(wandb.config)
        wandb.finish()
        wandb_setup._setup(_reset=True)

        config["wandb"]["sweep_parent"] = config["learn"]["run_name"]

        scores = []
        for i in range(num_folds):
            reset_wandb_env()
            ref_config["dataset_fold"] = i
            final_score = train(config, ref_config)
            if final_score is not None:
                scores.append(final_score)

        wandb.init(id=sweep_run_id, resume="must")
        wandb.log({sweep_config["metric"]["name"]: mean(scores)})
        wandb.finish()

    wandb.agent(
        sweep_config["sweep_id"],
        function=train_cv if use_cv else train,
        project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
        count=count,
    )


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fit-test", "fit", "test", "sweep", "sweep-cv"]),
    default="fit-test",
)
@click.option("--sweep_count", "-sc", type=int, default=3)
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
    mode: RunMode,
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    sweep_count: int,
    configs: list[tuple[str, str]],
):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")

    config = make_config(
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner="simple"
    )
    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    if mode == "fit-test":
        run_fit_test(config, dummy, resume=resume)
    elif mode == "fit":
        run_fit_test(config, dummy, resume=resume, fit_only=True)
    elif mode == "test":
        run_fit_test(config, dummy, resume=resume, test_only=True)
    elif mode == "sweep":
        run_sweep(config, dummy, count=sweep_count)
    elif mode == "sweep-cv":
        run_sweep(config, dummy, use_cv=True, count=sweep_count)


if __name__ == "__main__":
    main()
