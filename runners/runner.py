import os
from typing import Type

import optuna
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
import wandb.util
from config.config_maker import make_run_name
from config.config_type import ConfigUnion, OptunaPruner, OptunaSampler
from config.constants import FILENAMES, WANDB_SETTINGS
from config.optuna import OptunaConfig, default_optuna_config
from learners.base_learner import BaseLearner
from runners.callbacks import CustomRichProgressBar, custom_rich_progress_bar_theme
from utils.logging import (
    dump_json,
    get_configuration,
    get_full_ckpt_path,
    get_optuna_db_path,
)
from utils.utils import mean
from utils.wandb import (
    prepare_study_artifact_name,
    wandb_delete_file,
    wandb_download_file,
    wandb_log_file,
    wandb_login,
)


class Runner:
    def __init__(
        self,
        config: ConfigUnion,
        dummy: bool,
        resume: bool = False,
    ):
        self.config = config
        self.dummy = dummy
        self.resume = resume

        self.use_wandb = self.config.get("wandb") is not None
        self.exp_name = self.config["learn"]["exp_name"]
        self.run_name = self.config["learn"]["run_name"]

        self.optuna_config = self.make_optuna_config()

        self.sampler_classes: dict[OptunaSampler, Type[optuna.samplers.BaseSampler]] = {
            "random": optuna.samplers.RandomSampler,
            "tpe": optuna.samplers.TPESampler,
            "cmaes": optuna.samplers.CmaEsSampler,
            "qmc": optuna.samplers.QMCSampler,
            "gp": optuna.samplers.GPSampler,
        }
        self.pruner_classes: dict[OptunaPruner, Type[optuna.pruners.BasePruner]] = {
            "none": optuna.pruners.NopPruner,
            "median": optuna.pruners.MedianPruner,
            "percentile": optuna.pruners.PercentilePruner,
            "asha": optuna.pruners.SuccessiveHalvingPruner,
            "hyperband": optuna.pruners.HyperbandPruner,
            "threshold": optuna.pruners.ThresholdPruner,
        }

    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        learner_ckpt: str | None = None,
        optuna_trial: optuna.Trial | None = None,
    ) -> BaseLearner:
        raise NotImplementedError

    def make_optuna_config(self) -> OptunaConfig:
        raise NotImplementedError

    def update_trial_config(self, trial: optuna.Trial, config: ConfigUnion):
        raise NotImplementedError

    def make_trainer(self, **kwargs) -> Trainer:
        callbacks = self.make_callbacks()
        default_kwargs: dict = {"num_sanity_val_steps": 0}
        return Trainer(
            max_epochs=self.config["learn"]["num_epochs"],
            check_val_every_n_epoch=self.config["learn"].get("val_freq", 1),
            callbacks=callbacks,
            logger=False,
            inference_mode=not self.config["learn"].get("manual_optim", False),
            **(default_kwargs | kwargs),
        )

    def run_fit_test(
        self,
        fit_only: bool = False,
        test_only: bool = False,
    ):
        if self.use_wandb:
            wandb_login()
            if self.resume:
                prev_config = get_configuration(self.exp_name, self.run_name)
                run_id = prev_config["config"]["wandb"]["run_id"]
            else:
                run_id = wandb.util.generate_id()
            self.wandb_init(run_id)
            assert "wandb" in self.config
            self.config["wandb"]["run_id"] = run_id

        ref_ckpt_path = self.config["learn"].get("ref_ckpt_path")
        if (self.resume and not test_only) or (test_only and ref_ckpt_path is None):
            ckpt_path = get_full_ckpt_path(self.exp_name, self.run_name, "last.ckpt")
        else:
            ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

        trainer = self.make_trainer()
        learner = self.make_learner(self.config, self.dummy, learner_ckpt=ckpt_path)
        if not learner.init(resume=self.resume, force_clear_dir=True):
            return

        if not test_only:
            trainer.fit(learner, ckpt_path=ckpt_path if self.resume else None)

        if not fit_only:
            if not test_only:
                assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
                ckpt_path = trainer.checkpoint_callback.best_model_path
                trainer = self.make_trainer()
            trainer.test(learner, ckpt_path=ckpt_path)

        if self.use_wandb:
            wandb.finish()

    def run_study(self):
        assert self.optuna_config is not None

        def objective(trial: optuna.Trial) -> float:
            assert self.optuna_config is not None

            scores = []
            trial_config = self.config
            self.update_trial_config(trial, trial_config)
            trial_config["learn"]["run_name"] = make_run_name()
            trial_config["learn"]["optuna_study_name"] = self.optuna_config[
                "study_name"
            ]

            score = self.fit_study(trial, trial_config, 0)
            if score is not None:
                scores.append(score)

            for fold in range(1, self.optuna_config.get("num_folds", 1)):
                trial_config["learn"]["run_name"] += f" F{fold}"
                score = self.fit_study(None, trial_config, fold)
                if score is not None:
                    scores.append(score)

            return mean(scores)

        used_optuna_config = {**default_optuna_config, **self.optuna_config}

        sampler_class = self.sampler_classes[used_optuna_config["sampler"]]
        pruner_class = self.pruner_classes[used_optuna_config["pruner"]]

        optuna_db = get_optuna_db_path(self.dummy, self.config.get("wandb") is not None)
        if used_optuna_config["study_name"] == "":
            exp_name = self.config["learn"]["exp_name"]
            used_optuna_config["study_name"] = f"{exp_name} {make_run_name()}"

        wandb_download_file(optuna_db.split(".")[0], "", "optuna", self.dummy)

        study_kwargs = {
            "study_name": used_optuna_config["study_name"],
            "storage": f"sqlite:///{optuna_db}",
            "sampler": sampler_class(**used_optuna_config["sampler_params"]),
            "pruner": pruner_class(**used_optuna_config["pruner_params"]),
        }

        if self.resume:
            study = optuna.load_study(**study_kwargs)
        else:
            study = optuna.create_study(
                direction=used_optuna_config["direction"], **study_kwargs
            )

        for key, value in self.optuna_config.items():
            study.set_user_attr(key, value)

        study.optimize(
            objective,
            n_trials=used_optuna_config.get("num_trials"),
            timeout=used_optuna_config.get("timeout_sec", 3600),
            gc_after_trial=True,
        )

    def fit_study(
        self,
        trial: optuna.Trial | None,
        trial_config: ConfigUnion,
        dataset_fold: int,
    ) -> float | None:
        learner = self.make_learner(
            trial_config,
            self.dummy,
            dataset_fold=dataset_fold,
            optuna_trial=trial,
        )

        if self.use_wandb and not self.resume and trial and trial.number == 0:
            study_name = trial.study.study_name
            exp_name, run_name = study_name.split(" ", 1)

            wandb_login()
            wandb.init(
                tags=["helper"],
                project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
                group=exp_name,
                name=f"study {run_name}",
                job_type="study",
            )

            ref_configuration = learner.configuration
            ref_configuration["optuna"] = self.optuna_config
            ref_conf_path = os.path.join(
                FILENAMES["log_folder"], exp_name, f"study {run_name}.json"
            )
            dump_json(ref_conf_path, ref_configuration)

            wandb_log_file(
                wandb.run,
                prepare_study_artifact_name(study_name),
                ref_conf_path,
                "study",
            )

            wandb.finish()

        if self.use_wandb:
            run_id = wandb.util.generate_id()
            self.wandb_init(run_id)

        trainer = self.make_trainer()
        learner.init()
        trainer.fit(learner)

        if self.use_wandb:
            if trial is not None:
                optuna_db = get_optuna_db_path(self.dummy, True)
                artifact_name = optuna_db.split(".")[0]
                wandb_log_file(wandb.run, artifact_name, optuna_db, "optuna")
                wandb_delete_file(artifact_name, "optuna", False)
            wandb.finish()

        assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
        best_score = trainer.checkpoint_callback.best_model_score
        return best_score if best_score is None else best_score.item()

    def make_callbacks(self) -> list[Callback]:
        callbacks = []
        cb_config = self.config["callbacks"]

        if cb_config.get("progress"):
            callbacks.append(
                CustomRichProgressBar(
                    leave=cb_config.get("progress_leave", False),
                    theme=custom_rich_progress_bar_theme,
                )
            )

        monitor = cb_config.get("monitor", None)
        monitor_mode = cb_config.get("monitor_mode", "min")

        save_last = cb_config.get("ckpt_last", False)
        save_top_k = cb_config.get("ckpt_top_k", 1)
        ckpt_path = os.path.join(
            FILENAMES["checkpoint_folder"], self.exp_name, self.run_name
        )
        ckpt_filename = ("{epoch} {" + monitor + ":.2f}") if monitor else ("{epoch}")
        if save_last or save_top_k:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_path,
                    filename=ckpt_filename,
                    monitor=monitor,
                    mode=monitor_mode,
                    save_last=cb_config.get("ckpt_last", True),
                    save_top_k=cb_config.get("ckpt_top_k", 0),
                    every_n_epochs=self.config["learn"].get("val_freq", 1),
                )
            )

        if monitor:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=monitor_mode,
                    verbose=True,
                    patience=cb_config.get("stop_patience", 3),
                    min_delta=cb_config.get("stop_min_delta", 0.0),
                    stopping_threshold=cb_config.get("stop_threshold", None),
                )
            )

        return callbacks

    def wandb_init(self, run_id: str):
        assert "wandb" in self.config
        wandb.init(
            config=dict(self.config),
            id=run_id,
            tags=self.config["wandb"]["tags"],
            project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
            group=self.config["learn"]["exp_name"],
            name=self.config["learn"]["run_name"],
            job_type=self.config["wandb"]["job_type"],
            resume="must" if self.resume else None,
        )
