import click
from dotenv import load_dotenv

from config.config_maker import make_config
from config.config_type import (
    LearnerType,
    RunMode,
    TaskType,
    learner_types,
    run_modes,
    task_types,
)
from config.optuna import OptunaConfig
from tasks.optic_disc_cup.datasets import NUM_CLASSES as NUM_CLASSES_OPTIC
from tasks.optic_disc_cup.runners import get_runner_class as get_runner_class_optic
from tasks.skin_lesion.datasets import NUM_CLASSES as NUM_CLASSES_SKIN
from tasks.skin_lesion.runners import get_runner_class as get_runner_class_skin
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
    "--task",
    "-t",
    type=click.Choice(task_types),
    default="optic",
)
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
    "--configs",
    "-c",
    nargs=2,
    multiple=True,
    type=(str, str),
    default=[],
    help="(key, value) for overriding config, use '/' for nesting keys",
)
@click.option(
    "--options",
    "-o",
    nargs=2,
    multiple=True,
    type=(str, str),
    default=[],
    help="(key, value) for minor options, there are number_of_multi (int) and dataset_fold (int)",
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
    task: TaskType,
    learner: LearnerType,
    mode: RunMode,
    dataset: str,
    dummy: bool,
    resume: bool,
    no_wandb: bool,
    configs: list[tuple[str, str]],
    options: list[tuple[str, str]],
    optuna_configs: list[tuple[str, str]],
):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")

    if task == "optic":
        get_runner_class = get_runner_class_optic
        NUM_CLASSES = NUM_CLASSES_OPTIC
    elif task == "skin":
        get_runner_class = get_runner_class_skin
        NUM_CLASSES = NUM_CLASSES_SKIN
    else:
        raise ValueError(f"Unknown task: {task}")

    load_dotenv()

    config = make_config(
        mode=mode, dummy=dummy, use_wandb=not no_wandb, learner=learner
    )
    config["data"]["num_classes"] = NUM_CLASSES

    for key, value in configs:
        [parent_key, child_key] = key.split("/")
        config[parent_key][child_key] = parse_string(value)

    options_dict = dict(options)
    number_of_multi = int(options_dict.get("number_of_multi", 0))
    dataset_fold = int(options_dict.get("dataset_fold", 0))
    optuna_seed = (
        int(options_dict["optuna_seed"]) if "optuna_seed" in options_dict else None
    )

    optuna_config: OptunaConfig = {}
    for key, value in optuna_configs:
        if key == "hyperparams":
            optuna_config[key] = parse_hyperparams(value)
            continue
        optuna_config[key] = parse_string(value)
    if optuna_seed is not None:
        optuna_config["seed"] = optuna_seed

    runner_class = get_runner_class(learner)

    runner = runner_class(
        config,
        optuna_config,
        mode,
        task,
        learner,
        dummy,
        dataset=dataset,
        resume=resume,
    )

    if number_of_multi > 0:
        runner.number_of_multi = number_of_multi

    if mode in ["fit-test", "fit", "test"]:
        with wandb_use_alert():
            runner.run_fit_test(mode == "fit", mode == "test", dataset_fold)
        return

    if mode == "profile-fit":
        with wandb_use_alert():
            runner.run_multi_fit_test(True, False, dataset_fold)
        return
    if mode == "profile-test":
        with wandb_use_alert():
            runner.run_multi_fit_test(False, True, dataset_fold)
        return

    if mode == "study":
        with wandb_use_alert():
            runner.run_study()


if __name__ == "__main__":
    main()
