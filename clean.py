from typing import Literal

import click
from dotenv import load_dotenv

from config.config_type import TaskType, task_types
from runners.cleaners import clean_local_wandb, clean_logging_data


@click.command()
@click.option(
    "--limit",
    "-l",
    type=str,
    default="60",
    help="Time limit in last minutes (int) or timestamp (str). Will be used as start time, except when local_wandb is True.",
)
@click.option(
    "--dummy_only",
    "-d",
    is_flag=True,
)
@click.option(
    "--local_wandb",
    "-lw",
    is_flag=True,
    help="If True, will clean local wandb folders instead of logging data.",
)
@click.option(
    "--task",
    "-t",
    type=click.Choice(task_types),
    default="optic",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["local", "wandb", "both"]),
    default="both",
    help="If local_wandb is False, this determine which logging data to clean.",
)
@click.option(
    "--force_clean",
    "-f",
    is_flag=True,
)
def main(
    limit: str,
    dummy_only: bool,
    local_wandb: bool,
    task: TaskType,
    mode: Literal["local", "wandb", "both"],
    force_clean: bool,
):
    try:
        new_limit = int(limit)
    except ValueError:
        new_limit = limit

    load_dotenv()

    if local_wandb:
        clean_local_wandb(new_limit, force_clean)
    else:
        clean_logging_data(new_limit, task, mode, dummy_only, force_clean)


if __name__ == "__main__":
    main()
