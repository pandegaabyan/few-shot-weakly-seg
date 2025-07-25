import click

from config.config_maker import make_config
from config.config_type import LearnerType, RunMode, learner_types, run_modes
from tasks.optic_disc_cup.runners import get_runner_class
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
    config["data"]["num_classes"] = 3

    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    runner_class = get_runner_class(learner)

    runner = runner_class(config, mode, learner, dummy, dataset=dataset, resume=resume)

    if number_of_multi > 0:
        runner.number_of_multi = number_of_multi

    for key, value in optuna_configs:
        if key == "hyperparams":
            runner.optuna_config[key] = parse_hyperparams(value)
            continue
        runner.optuna_config[key] = parse_string(value)

    if mode in ["fit-test", "fit", "test"]:
        with wandb_use_alert():
            runner.run_fit_test(mode == "fit", mode == "test")
        return

    if mode == "profile-fit":
        with wandb_use_alert():
            runner.run_multi_fit_test(True, False)
        return
    if mode == "profile-test":
        with wandb_use_alert():
            runner.run_multi_fit_test(False, True)
        return

    if mode == "study":
        with wandb_use_alert():
            runner.run_study()


if __name__ == "__main__":
    main()
