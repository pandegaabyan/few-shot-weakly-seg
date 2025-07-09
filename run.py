import click

from config.config_maker import make_config
from config.config_type import LearnerType, RunMode, learner_types, run_modes
from my_runners import ProtosegRunner, SimpleRunner, WeaselRunner
from utils.logging import (
    check_git_clean,
)
from utils.optuna import parse_hyperparams
from utils.utils import parse_string
from utils.wandb import wandb_use_alert


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--learner",
    "-l",
    type=click.Choice(learner_types),
    default="SL",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(run_modes),
    default="fit-test",
)
@click.option(
    "--dataset",
    "-ds",
    type=str,
    default="all",
)
@click.option(
    "--number_of_multi",
    "-num",
    type=int,
    default=-1,
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
    learner: LearnerType,
    mode: RunMode,
    dataset: str,
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    number_of_multi: int,
    configs: list[tuple[str, str]],
    optuna_configs: list[tuple[str, str]],
):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")

    config = make_config(
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner=learner
    )

    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    if mode == "fit-test" and "wandb" in config:
        config["wandb"].update(
            {
                "tags": ["var_region"],
                "watch_model": False,
                "save_model": False,
                "save_train_preds": 0,
                "save_val_preds": 0,
                "save_test_preds": 0,
            }
        )

    runner_name = learner.split("-")[0]
    if runner_name == "SL":
        runner_class = SimpleRunner
    elif runner_name == "WS":
        runner_class = WeaselRunner
    elif runner_name == "PS":
        runner_class = ProtosegRunner

    runner = runner_class(config, mode, learner, dummy, dataset=dataset, resume=resume)

    if number_of_multi > 0:
        runner.number_of_multi = number_of_multi

    for key, value in optuna_configs:
        if key == "hyperparams":
            runner.optuna_config[key] = parse_hyperparams(value)
            continue
        runner.optuna_config[key] = parse_string(value)

    if mode == "fit-test":
        with wandb_use_alert():
            runner.run_multi_fit_test(False, False)
        return

    if mode == "profile-test":
        with wandb_use_alert():
            runner.run_multi_fit_test(False, True)


if __name__ == "__main__":
    main()
