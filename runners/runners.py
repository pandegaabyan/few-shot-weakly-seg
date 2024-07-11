from typing import Callable, Protocol, Type

import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import wandb
import wandb.util
from config.config_maker import make_run_name
from config.config_type import ConfigUnion, OptunaPruner, OptunaSampler
from config.constants import WANDB_SETTINGS
from config.optuna import OptunaConfig, default_optuna_config
from learners.base_learner import BaseLearner
from runners.trainer import make_trainer
from utils.logging import get_configuration, get_full_ckpt_path, get_optuna_db_path
from utils.utils import mean
from utils.wandb import wandb_download_file, wandb_log_file, wandb_login


class MakeLearnerTrainer(Protocol):
    def __call__(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = ...,
        learner_ckpt: str | None = ...,
        optuna_trial: optuna.Trial | None = ...,
    ) -> tuple[BaseLearner, Trainer]:
        ...


def run_fit_test(
    config: ConfigUnion,
    dummy: bool,
    make_learner_and_trainer: MakeLearnerTrainer,
    resume: bool = False,
    fit_only: bool = False,
    test_only: bool = False,
    dataset_fold: int = 0,
    optuna_trial: optuna.Trial | None = None,
) -> float | None:
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
            run_id = wandb.util.generate_id()
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
        config,
        dummy,
        learner_ckpt=ckpt_path,
        dataset_fold=dataset_fold,
        optuna_trial=optuna_trial,
    )
    if not learner.init(resume=resume, force_clear_dir=True):
        return

    if not test_only:
        trainer.fit(learner, ckpt_path=ckpt_path if resume else None)

    if not fit_only:
        if not test_only:
            assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
            ckpt_path = trainer.checkpoint_callback.best_model_path
            trainer = make_trainer(config)
        trainer.test(learner, ckpt_path=ckpt_path)

    if use_wandb:
        if optuna_trial is not None:
            optuna_db = get_optuna_db_path(dummy, True)
            wandb_log_file(wandb.run, optuna_db.split(".")[0], optuna_db, "optuna")
        wandb.finish()

    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
    best_score = trainer.checkpoint_callback.best_model_score
    return best_score if best_score is None else best_score.item()


def run_study(
    config: ConfigUnion,
    optuna_config: OptunaConfig,
    dummy: bool,
    make_learner_and_trainer: MakeLearnerTrainer,
    update_trial_config: Callable[[optuna.Trial, ConfigUnion], None],
    resume: bool = False,
):
    def objective(trial: optuna.Trial) -> float:
        scores = []
        trial_config = config
        update_trial_config(trial, trial_config)
        trial_config["learn"]["run_name"] = make_run_name()
        trial_config["learn"]["optuna_study_name"] = optuna_config["study_name"]

        fit_test_kwargs = {
            "config": trial_config,
            "dummy": dummy,
            "make_learner_and_trainer": make_learner_and_trainer,
            "resume": resume,
            "fit_only": True,
        }

        score = run_fit_test(**fit_test_kwargs, dataset_fold=0, optuna_trial=trial)
        if score is not None:
            scores.append(score)

        for fold in range(1, optuna_config.get("num_folds", 1)):
            fit_test_kwargs["config"]["learn"]["run_name"] += f" F{fold}"
            score = run_fit_test(**fit_test_kwargs, dataset_fold=fold)
            if score is not None:
                scores.append(score)

        return mean(scores)

    used_optuna_config = {**default_optuna_config, **optuna_config}

    sampler_classes: dict[OptunaSampler, Type[optuna.samplers.BaseSampler]] = {
        "random": optuna.samplers.RandomSampler,
        "tpe": optuna.samplers.TPESampler,
        "cmaes": optuna.samplers.CmaEsSampler,
        "qmc": optuna.samplers.QMCSampler,
        "gp": optuna.samplers.GPSampler,
    }
    pruner_classes: dict[OptunaPruner, Type[optuna.pruners.BasePruner]] = {
        "none": optuna.pruners.NopPruner,
        "median": optuna.pruners.MedianPruner,
        "percentile": optuna.pruners.PercentilePruner,
        "asha": optuna.pruners.SuccessiveHalvingPruner,
        "hyperband": optuna.pruners.HyperbandPruner,
        "threshold": optuna.pruners.ThresholdPruner,
    }

    sampler_class = sampler_classes[used_optuna_config["sampler"]]
    pruner_class = pruner_classes[used_optuna_config["pruner"]]

    optuna_db = get_optuna_db_path(dummy, config.get("wandb") is not None)
    if used_optuna_config["study_name"] == "":
        exp_name = config["learn"]["exp_name"]
        used_optuna_config["study_name"] = f"{exp_name} {make_run_name()}"

    wandb_download_file(optuna_db.split(".")[0], "", "optuna", dummy)

    study_kwargs = {
        "study_name": used_optuna_config["study_name"],
        "storage": f"sqlite:///{optuna_db}",
        "sampler": sampler_class(**used_optuna_config["sampler_params"]),
        "pruner": pruner_class(**used_optuna_config["pruner_params"]),
    }

    if resume:
        study = optuna.load_study(**study_kwargs)
    else:
        study = optuna.create_study(
            direction=used_optuna_config["direction"], **study_kwargs
        )

    for key, value in optuna_config.items():
        study.set_user_attr(key, value)

    study.optimize(
        objective,
        n_trials=used_optuna_config.get("num_trials"),
        timeout=used_optuna_config.get("timeout_sec"),
        gc_after_trial=True,
    )
