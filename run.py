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
    config["data"]["batch_size"] = 16
    config["learn"]["num_epochs"] = 300
    config["callbacks"]["stop_patience"] = 15
    config["optimizer"]["betas"] = (0.9, 0.999)
    config["optimizer"]["lr"] = 0.00015
    config["optimizer"]["lr_bias_mult"] = 2
    config["optimizer"]["weight_decay"] = 0.0053
    config["scheduler"]["gamma"] = 0.24731
    config["scheduler"]["step_size"] = 50

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

    for key, value in optuna_configs:
        runner.optuna_config[key] = parse_string(value)

    if mode in ["study"]:
        runner.run_study()


if __name__ == "__main__":
    main()
