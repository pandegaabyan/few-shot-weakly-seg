from typing import Type

import click

from config.config_maker import make_config
from config.config_type import LearnerType, RunMode
from my_runners import ProtosegRunner, SimpleRunner, WeaselRunner
from runners.runner import Runner
from utils.logging import (
    check_git_clean,
)
from utils.utils import parse_string


@click.command()
@click.option("--dummy", "-d", is_flag=True)
@click.option("--resume", "-r", is_flag=True)
@click.option("--no_wandb", "-nw", is_flag=True)
@click.option(
    "--learner",
    "-l",
    type=click.Choice(["simple", "weasel", "protoseg", "guidednets"]),
    default="simple",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(
        ["fit-test", "fit", "test", "study", "profile-fit", "profile-test"]
    ),
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
    learner: LearnerType,
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
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner=learner
    )

    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    runner_classes: dict[LearnerType, Type[Runner]] = {
        "simple": SimpleRunner,
        "weasel": WeaselRunner,
        "protoseg": ProtosegRunner,
    }
    runner = runner_classes[learner](config, dummy, resume)

    if mode in ["fit-test", "fit", "test"]:
        runner.run_fit_test(mode == "fit", mode == "test")
        return

    if mode == "profile-fit":
        runner.run_profile("fit")
        return
    if mode == "profile-test":
        runner.run_profile("test")
        return

    for key, value in optuna_configs:
        runner.optuna_config[key] = parse_string(value)

    if mode == "study":
        runner.run_study()


if __name__ == "__main__":
    main()
