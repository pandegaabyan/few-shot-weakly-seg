import os
from typing import Sequence, Type

import nanoid
import optuna
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
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
from data.base_dataset import BaseDataset
from data.typings import BaseDatasetKwargs
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
    prepare_study_artifact_name,
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
        deterministic = self.config["learn"].get("deterministic", True)
        progress = self.config["callbacks"].get("progress", True)
        return Trainer(
            max_epochs=self.config["learn"]["num_epochs"],
            check_val_every_n_epoch=self.config["learn"].get("val_freq", 1),
            callbacks=callbacks,
            deterministic="warn" if deterministic else False,
            benchmark=not deterministic,
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
            trial_config = self.config
            run_name = make_run_name()
            self.run_name = run_name
            trial_config["learn"]["run_name"] = run_name
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

        sampler_class = sampler_classes[self.optuna_config["sampler"]]
        pruner_class = pruner_classes[self.optuna_config["pruner"]]

        if not self.resume:
            self.optuna_config["study_name"] += f" {nanoid.generate(size=5)}"
            self.optuna_config["study_name"] = self.optuna_config["study_name"].strip()

        study_kwargs = {
            "study_name": self.optuna_config["study_name"],
            "storage": get_optuna_storage(self.dummy),
            "sampler": sampler_class(**self.optuna_config.get("sampler_params", {})),
            "pruner": optuna.pruners.PatientPruner(
                pruner_class(**self.optuna_config.get("pruner_params")),
                self.optuna_config.get("pruner_patience", 1),
            ),
        }

        if self.resume:
            study = optuna.load_study(**study_kwargs)
        else:
            study = optuna.create_study(
                direction=self.optuna_config["direction"], **study_kwargs
            )
            study.set_user_attr("git_hash", get_short_git_hash())
            for key, value in self.optuna_config.items():
                if key == "study_name":
                    continue
                study.set_user_attr(key, value)

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
        trial_config: ConfigUnion,
        dataset_fold: int,
    ) -> float | None:
        if self.use_wandb:
            run_id = wandb.util.generate_id()
            assert "wandb" in trial_config
            trial_config["wandb"]["run_id"] = run_id

        learner_class, learner_kwargs, important_config = self.make_learner(
            trial_config,
            self.dummy,
            dataset_fold=dataset_fold,
            optuna_trial=trial,
        )
        learner = learner_class(**learner_kwargs)

        if self.use_wandb and not self.resume and trial and trial.number == 0:
            study_id = trial.study.study_name.split(" ")[-1]

            wandb_login()
            wandb.init(
                tags=["helper"],
                project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
                group=self.exp_name,
                name=f"log study-ref {study_id}",
                job_type="study",
            )

            ref_configuration = learner.get_configuration()
            ref_configuration["optuna"] = self.optuna_config
            exp_path = os.path.join(FILENAMES["log_folder"], self.exp_name)
            check_mkdir(exp_path)
            ref_conf_path = os.path.join(exp_path, f"{study_id} study-ref.json")
            dump_json(ref_conf_path, ref_configuration)

            wandb_log_file(
                wandb.run,
                prepare_study_artifact_name(study_id),
                ref_conf_path,
                "study-reference",
            )

            wandb.finish()

        if self.use_wandb:
            self.wandb_init(run_id)
            study_name = trial_config["learn"].get("optuna_study_name")
            assert study_name is not None
            wandb.config.update(
                {
                    "git": get_short_git_hash(),
                    "study": study_name.split(" ")[-1],
                    **important_config,
                }
            )
        learner.init()

        trainer = self.make_trainer()
        trainer.fit(learner)

        if self.use_wandb:
            wandb.finish()

        if learner.optuna_pruned:
            raise optuna.TrialPruned()

        return learner.best_monitor_value

    def make_callbacks(self) -> list[Callback]:
        callbacks = []
        cb_config = self.config["callbacks"]

        monitor = cb_config.get("monitor")
        monitor_mode = cb_config.get("monitor_mode", "min")

        log_path = os.path.join(FILENAMES["log_folder"], self.exp_name, self.run_name)
        ckpt_filename = ("{epoch} {" + monitor + ":.2f}") if monitor else ("{epoch}")
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
        wandb.init(
            id=run_id,
            tags=self.config["wandb"].get("tags", []),
            project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
            group=self.config["learn"]["exp_name"],
            name=self.config["learn"]["run_name"],
            job_type=self.config["wandb"].get("job_type"),
            resume="must" if resume else None,
        )

    def get_names_from_dataset_list(
        self, dataset_list: Sequence[tuple[Type[BaseDataset], BaseDatasetKwargs]]
    ) -> str:
        return ",".join(
            [(kwargs.get("dataset_name") or "NoName") for _, kwargs in dataset_list]
        )
