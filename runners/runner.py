import os
from typing import Type

import optuna
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
import wandb.errors
import wandb.util
from config.config_maker import make_run_name
from config.config_type import ConfigUnion
from config.constants import FILENAMES, WANDB_SETTINGS
from config.optuna import (
    OptunaConfig,
    default_optuna_config,
    get_optuna_storage,
    pruner_classes,
    sampler_classes,
)
from learners.base_learner import BaseLearner
from learners.typings import BaseLearnerKwargs
from runners.callbacks import CustomRichProgressBar, custom_rich_progress_bar_theme
from utils.logging import (
    check_mkdir,
    dump_json,
    get_full_ckpt_path,
    get_short_git_hash,
)
from utils.utils import mean
from utils.wandb import (
    prepare_ckpt_artifact_alias,
    prepare_study_ckpt_artifact_name,
    prepare_study_ref_artifact_name,
    wandb_delete_file,
    wandb_download_ckpt,
    wandb_download_config,
    wandb_get_run_id_by_name,
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
        self.dummy = dummy
        self.resume = resume

        self.config = self.update_config(config)
        self.optuna_config = self.make_optuna_config()

        self.use_wandb = self.config.get("wandb") is not None
        self.exp_name = self.config["learn"]["exp_name"]
        self.run_name = self.config["learn"]["run_name"]

        self.curr_trial_number = -1
        self.curr_dataset_fold = -1

    def make_learner(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[BaseLearner], BaseLearnerKwargs, dict]:
        raise NotImplementedError

    def update_config(self, config: ConfigUnion) -> ConfigUnion:
        return config

    def make_optuna_config(self) -> OptunaConfig:
        return default_optuna_config

    def make_trainer(self, **kwargs) -> Trainer:
        callbacks = self.make_callbacks()
        progress = self.config["callbacks"].get("progress", True)
        return Trainer(
            max_epochs=self.config["learn"]["num_epochs"],
            check_val_every_n_epoch=self.config["learn"].get("val_freq", 1),
            callbacks=callbacks,
            deterministic=self.config["learn"].get("cudnn_deterministic", "warn"),
            benchmark=self.config["learn"].get("cudnn_benchmark", False),
            logger=False,
            enable_progress_bar=progress,
            enable_model_summary=progress,
            inference_mode=not self.config["learn"].get("manual_optim", False),
            **kwargs,
        )

    def run_fit_test(
        self,
        fit_only: bool = False,
        test_only: bool = False,
    ):
        ref_ckpt_path = self.config["learn"].get("ref_ckpt_path")
        test_only_by_resuming = test_only and ref_ckpt_path is None
        if (self.resume and not test_only) or test_only_by_resuming:
            ckpt_path = get_full_ckpt_path(self.exp_name, self.run_name, "last.ckpt")
        else:
            ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

        if self.use_wandb:
            wandb_login()
            if self.resume or test_only_by_resuming:
                run_id = wandb_get_run_id_by_name(self.run_name, dummy=self.dummy)
            else:
                run_id = wandb.util.generate_id()
            assert "wandb" in self.config
            self.config["wandb"]["run_id"] = run_id
            self.wandb_init(run_id, resume=self.resume or test_only_by_resuming)
            if ckpt_path is not None:
                wandb_download_ckpt(ckpt_path)
            if self.resume or test_only_by_resuming:
                wandb_download_config(self.exp_name, self.run_name)

        learner_class, learner_kwargs, important_config = self.make_learner(
            self.config, self.dummy
        )
        if ckpt_path is None:
            learner = learner_class(**learner_kwargs)
        else:
            learner = learner_class.load_from_checkpoint(ckpt_path, **learner_kwargs)

        if self.use_wandb:
            wandb.config.update({"git": get_short_git_hash(), **important_config})
        init_ok = learner.init(
            resume=self.resume or test_only_by_resuming, force_clear_dir=True
        )
        if not init_ok:
            return

        trainer = self.make_trainer()
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
        def objective(trial: optuna.Trial) -> float:
            scores = []
            base_run_name = make_run_name()
            self.run_name = base_run_name
            self.config["learn"]["run_name"] = base_run_name
            study_id = self.optuna_config["study_name"].split(" ")[-1]
            self.config["learn"]["optuna_study"] = study_id

            trial.set_user_attr("run_name", base_run_name)
            self.curr_trial_number = trial.number
            self.curr_dataset_fold = 0

            score, pruned = self.fit_study(trial)
            base_wandb_run_id = self.config.get("wandb", {}).get("run_id")
            if pruned:
                self.wandb_log_trial(base_wandb_run_id, trial, score, pruned)
                raise optuna.TrialPruned()
            if score is not None:
                scores.append(score)

            for fold in range(1, self.optuna_config.get("num_folds", 1)):
                self.curr_dataset_fold = fold
                new_run_name = base_run_name + f" F{fold}"
                self.run_name = new_run_name
                self.config["learn"]["run_name"] = new_run_name
                score, _ = self.fit_study(None)
                if score is not None:
                    scores.append(score)

            new_score = mean(scores)
            self.wandb_log_trial(base_wandb_run_id, trial, new_score, False)

            return new_score

        sampler_class = sampler_classes[self.optuna_config["sampler"]]
        pruner_class = pruner_classes[self.optuna_config["pruner"]]

        pruner = pruner_class(**self.optuna_config.get("pruner_params", {}))
        pruner_patience = self.optuna_config.get("pruner_patience")
        if pruner_patience:
            pruner = optuna.pruners.PatientPruner(pruner, pruner_patience)
        study_kwargs = {
            "study_name": self.optuna_config["study_name"],
            "storage": get_optuna_storage(self.dummy),
            "sampler": sampler_class(**self.optuna_config.get("sampler_params", {})),
            "pruner": pruner,
        }

        try:
            study = optuna.create_study(
                direction=self.optuna_config["direction"], **study_kwargs
            )
            study.set_user_attr("git_hash", get_short_git_hash())
            for key, value in self.optuna_config.items():
                if key == "study_name":
                    continue
                study.set_user_attr(key, value)
        except optuna.exceptions.DuplicatedStudyError:
            study = optuna.load_study(**study_kwargs)

        n_trials = self.optuna_config.get("num_trials")
        timeout = self.optuna_config.get("timeout_sec")
        if n_trials is None and timeout is None:
            timeout = 120
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
        )

    def fit_study(
        self,
        trial: optuna.Trial | None,
    ) -> tuple[float, bool]:
        if self.use_wandb:
            run_id = wandb.util.generate_id()
            assert "wandb" in self.config
            self.config["wandb"]["run_id"] = run_id

        learner_class, learner_kwargs, important_config = self.make_learner(
            self.config,
            self.dummy,
            dataset_fold=self.curr_dataset_fold,
            optuna_trial=trial,
        )
        learner = learner_class(**learner_kwargs)

        study_id = self.optuna_config["study_name"].split(" ")[-1]
        additional_config: dict = {
            "git": get_short_git_hash(),
            "study": study_id,
        }

        if (
            self.use_wandb
            and self.curr_trial_number == 0
            and self.curr_dataset_fold == 0
        ):
            wandb_login()
            wandb.init(
                tags=["helper"],
                project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
                group=self.exp_name,
                name=f"log study-ref {study_id}",
                job_type="study",
            )
            wandb.config.update(additional_config)

            ref_configuration = learner.get_configuration()
            ref_configuration["optuna"] = self.optuna_config
            exp_path = os.path.join(FILENAMES["log_folder"], self.exp_name)
            check_mkdir(exp_path)
            ref_conf_path = os.path.join(exp_path, f"{study_id} study-ref.json")
            dump_json(ref_conf_path, ref_configuration)

            wandb_log_file(
                wandb.run,
                prepare_study_ref_artifact_name(study_id),
                ref_conf_path,
                "study-reference",
            )

            wandb.finish()

        if self.use_wandb:
            self.wandb_init(run_id)
            additional_config["trial"] = self.curr_trial_number
            if self.curr_dataset_fold > 0:
                additional_config["fold"] = self.curr_dataset_fold
            wandb.config.update(additional_config | important_config)
        learner.init()

        trainer = self.make_trainer()
        trainer.fit(learner)

        if self.use_wandb:
            wandb.finish()

        assert learner.best_monitor_value is not None
        return learner.best_monitor_value, learner.optuna_pruned

    def make_callbacks(self) -> list[Callback]:
        callbacks = []
        cb_config = self.config["callbacks"]

        monitor = cb_config.get("monitor")
        monitor_mode = cb_config.get("monitor_mode", "min")

        log_path = os.path.join(FILENAMES["log_folder"], self.exp_name, self.run_name)
        ckpt_filename = "{epoch}"
        if monitor:
            ckpt_filename = "{" + monitor + ":.4f} " + ckpt_filename
        if self.curr_trial_number != -1 and self.curr_dataset_fold != -1:
            ckpt_filename = (
                ckpt_filename
                + f" fold={self.curr_dataset_fold} trial={self.curr_trial_number}"
            )
        callbacks.append(
            ModelCheckpoint(
                dirpath=log_path,
                filename=ckpt_filename,
                monitor=monitor,
                mode=monitor_mode,
                save_last=cb_config.get("ckpt_last", False),
                save_top_k=cb_config.get("ckpt_top_k", 1),
                every_n_epochs=self.config["learn"].get("val_freq", 1),
            )
        )

        if monitor:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=monitor_mode,
                    patience=cb_config.get("stop_patience", 3),
                    min_delta=cb_config.get("stop_min_delta", 0.0),
                    stopping_threshold=cb_config.get("stop_threshold"),
                )
            )

        if cb_config.get("progress"):
            callbacks.append(
                CustomRichProgressBar(
                    leave=True,
                    theme=custom_rich_progress_bar_theme,
                )
            )

        return callbacks

    def wandb_init(self, run_id: str, resume: bool = False):
        assert "wandb" in self.config
        if resume:
            wandb.init(id=run_id, resume="must")
            return
        wandb.init(
            id=run_id,
            tags=self.config["wandb"].get("tags", []),
            project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
            group=self.config["learn"]["exp_name"],
            name=self.config["learn"]["run_name"],
            job_type=self.config["wandb"].get("job_type"),
        )

    def wandb_log_trial(
        self, run_id: str | None, trial: optuna.Trial, new_score: float, pruned: bool
    ):
        if run_id is None:
            return

        self.wandb_init(run_id, resume=True)

        wandb.log({"trial_score": new_score})

        if pruned or not self.config["callbacks"].get("ckpt_top_k"):
            wandb.finish()
            return

        minimize = self.optuna_config["direction"] == "minimize"
        try:
            best_score = trial.study.best_value
        except ValueError:
            best_score = float("inf") if minimize else -float("inf")
        if (minimize and new_score >= best_score) or (
            not minimize and new_score <= best_score
        ):
            wandb.finish()
            return

        study_id = self.optuna_config["study_name"].split(" ")[-1]
        artifact_name = prepare_study_ckpt_artifact_name(study_id)
        try:
            wandb_delete_file(
                artifact_name,
                "study-checkpoint",
                dummy=self.config["learn"].get("dummy") is True,
            )
        except (TypeError, wandb.errors.CommError):
            pass

        index = 0 if self.config["callbacks"].get("monitor_mode") == "min" else -1
        base_run_name = " ".join(self.config["learn"]["run_name"].split(" ")[:3])
        exp_path = os.path.join(FILENAMES["log_folder"], self.exp_name)
        run_paths = filter(
            lambda x: x.startswith(base_run_name),
            os.listdir(exp_path),
        )
        for run_path in run_paths:
            log_path = os.path.join(exp_path, run_path)
            best_ckpt = sorted(
                filter(
                    lambda x: x.endswith(".ckpt") and x != "last.ckpt",
                    os.listdir(log_path),
                )
            )[index]
            artifact_alias = prepare_ckpt_artifact_alias(best_ckpt)
            wandb_log_file(
                wandb.run,
                artifact_name,
                os.path.join(log_path, best_ckpt),
                "study-checkpoint",
                [artifact_alias],
            )

        wandb.finish()
