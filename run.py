import subprocess

import click
import nanoid
import optuna

from config.config_maker import make_config, make_exp_name
from config.config_type import AllConfig
from config.optuna import (
    OptunaConfig,
    get_optuna_storage,
    sampler_classes,
)
from data.get_meta_datasets import MetaDatasets, get_meta_datasets
from data.get_tune_loaders import TuneLoaderDict, get_tune_loaders
from learners.protoseg import ProtoSegLearner
from learners.weasel import WeaselLearner
from models.u_net import UNet
from tasks.optic_disc_cup.datasets import DrishtiDataset, RimOneDataset
from tasks.optic_disc_cup.metrics import calc_disc_cup_iou


def get_short_git_hash() -> str:
    short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    short_hash = str(short_hash, "utf-8").strip()
    return short_hash


def check_git_clean() -> bool:
    message = subprocess.check_output(["git", "status"])
    message = str(message, "utf-8").strip()
    return message.endswith("working tree clean")


def prepare_data(config: AllConfig) -> tuple[MetaDatasets, TuneLoaderDict]:
    rim_one_sparsity_params: dict = {
        "contour_radius_dist": 4,
        "contour_radius_thick": 2,
        "skeleton_radius_thick": 4,
        "region_compactness": 0.5,
    }
    meta_set = get_meta_datasets(
        [
            {
                "dataset_class": RimOneDataset,
                "num_classes": config["data"]["num_classes"],
                "resize_to": config["data"]["resize_to"],
                "kwargs": {
                    "split_seed": 0,
                    "split_test_size": 0.8,
                    "sparsity_mode": "random",
                    "sparsity_value": "random",
                    "sparsity_params": rim_one_sparsity_params,
                },
            }
        ]
    )

    drishti_sparsity_params: dict = {
        "contour_radius_dist": 4,
        "contour_radius_thick": 1,
        "skeleton_radius_thick": 3,
        "region_compactness": 0.5,
    }
    tune_loader = get_tune_loaders(
        dataset_class=DrishtiDataset,
        dataset_kwargs={
            "split_seed": 0,
            "split_test_size": 0.8,
            "sparsity_mode": "random",
            "sparsity_value": "random",
            "sparsity_params": drishti_sparsity_params,
        },
        num_classes=config["data"]["num_classes"],
        resize_to=config["data"]["resize_to"],
        shots=config["data_tune"]["list_shots"],
        point=config["data_tune"]["list_sparsity_point"],
        grid=config["data_tune"]["list_sparsity_grid"],
        contour=config["data_tune"]["list_sparsity_contour"],
        skeleton=config["data_tune"]["list_sparsity_skeleton"],
        region=config["data_tune"]["list_sparsity_region"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    return meta_set, tune_loader


def run_fit(config: AllConfig, learner_type: str) -> float:
    meta_set, tune_loader = prepare_data(config)

    if learner_type == "weasel":
        net = UNet(config["data"]["num_channels"], config["data"]["num_classes"])
        learner = WeaselLearner(net, config, meta_set, tune_loader, calc_disc_cup_iou)
    elif learner_type == "protoseg":
        net = UNet(config["data"]["num_channels"], config["protoseg"]["embedding_size"])
        learner = ProtoSegLearner(net, config, meta_set, tune_loader, calc_disc_cup_iou)
    else:
        raise ValueError("Invalid learner")

    learner.learn()

    return learner.best_avg_score


def run_study(config: AllConfig, learner_type: str, study_name: str, dummy: bool):
    resume = bool(study_name)
    if study_name == "":
        study_name = f"{learner_type} {nanoid.generate(size=5)}"
    optuna_config: OptunaConfig = {
        "study_name": study_name,
        "direction": "maximize",
        "sampler": "tpe",
        "timeout_sec": 20 * 60 if dummy else 8 * 3600,
        "sampler_params": {},
    }

    def objective(trial: optuna.Trial) -> float:
        config["save"]["exp_name"] = make_exp_name(learner=learner_type, dummy=dummy)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
        momentum_comp = trial.suggest_float("momentum_comp", 1e-2, 1, log=True)
        lowest_gamma = (1e-10 / lr) ** (
            config["learn"]["scheduler_step_size"] / config["learn"]["num_epochs"]
        )
        gamma = trial.suggest_float("gamma", lowest_gamma, 1, log=True)
        config["learn"]["optimizer_lr"] = lr
        config["learn"]["optimizer_weight_decay"] = weight_decay
        config["learn"]["optimizer_momentum"] = 1 - momentum_comp
        config["learn"]["scheduler_gamma"] = gamma

        if learner_type == "weasel":
            first_order = trial.suggest_categorical("first_order", [True, False])
            update_rate = trial.suggest_float("update_rate", 1e-2, 1, log=True)
            config["weasel"]["use_first_order"] = first_order
            config["weasel"]["update_param_rate"] = update_rate
        elif learner_type == "protoseg":
            embedding = trial.suggest_int("embedding", 2, 16)
            config["protoseg"]["embedding_size"] = embedding

        score = run_fit(config, learner_type)

        return score

    sampler_class = sampler_classes[optuna_config["sampler"]]
    study_kwargs = {
        "study_name": optuna_config["study_name"],
        "storage": get_optuna_storage(dummy),
        "sampler": sampler_class(**optuna_config["sampler_params"]),
    }

    if resume:
        study = optuna.load_study(**study_kwargs)
    else:
        study = optuna.create_study(
            direction=optuna_config["direction"], **study_kwargs
        )
        study.set_user_attr("git_hash", get_short_git_hash())
        for key, value in optuna_config.items():
            if key == "study_name":
                continue
            study.set_user_attr(key, value)

    study.optimize(
        objective,
        timeout=optuna_config["timeout_sec"],
        gc_after_trial=True,
    )


@click.command()
@click.option(
    "--learner",
    "-l",
    type=click.Choice(["weasel", "protoseg", ""]),
    default="",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fit", "study"]),
    default="fit",
)
@click.option(
    "--study_name",
    "-sn",
    type=str,
    default="",
)
@click.option("--dummy", "-d", is_flag=True)
def main(learner, mode, dummy, study_name):
    if not dummy and not check_git_clean():
        raise Exception("Git is not clean, please commit your changes first")
    print("Git hash:", get_short_git_hash())

    config = make_config(learner=learner or None, mode=mode, dummy=dummy)

    if mode == "fit":
        run_fit(config, learner)
    elif mode == "study":
        run_study(config, learner, study_name, dummy)


if __name__ == "__main__":
    main()
