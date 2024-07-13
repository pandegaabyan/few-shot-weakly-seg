import os

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
from learners.base_learner import BaseLearner
from runners.callbacks import CustomRichProgressBar, custom_rich_progress_bar_theme
from utils.logging import (
    dump_json,
    get_configuration,
    get_full_ckpt_path,
)
from utils.utils import mean
from utils.wandb import (
    prepare_study_artifact_name,
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
        ckpt_path: str | None = None,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[BaseLearner, dict]:
        raise NotImplementedError

    def update_config(self, config: ConfigUnion) -> ConfigUnion:
        return config

    def make_optuna_config(self) -> OptunaConfig:
        return default_optuna_config

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

        learner, important_config = self.make_learner(
            self.config, self.dummy, ckpt_path=ckpt_path
        )
        wandb.config.update(important_config)
        if not learner.init(resume=self.resume, force_clear_dir=True):
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

        sampler_class = sampler_classes[self.optuna_config["sampler"]]
        pruner_class = pruner_classes[self.optuna_config["pruner"]]

        if self.optuna_config["study_name"] == "":
            exp_name = self.config["learn"]["exp_name"]
            self.optuna_config["study_name"] = f"{exp_name} {make_run_name()}"

        study_kwargs = {
            "study_name": self.optuna_config["study_name"],
            "storage": get_optuna_storage(self.dummy),
            "sampler": sampler_class(**self.optuna_config.get("sampler_params", {})),
            "pruner": pruner_class(**self.optuna_config.get("pruner_params")),
        }

        if self.resume:
            study = optuna.load_study(**study_kwargs)
        else:
            study = optuna.create_study(
                direction=self.optuna_config["direction"], **study_kwargs
            )

        for key, value in self.optuna_config.items():
            study.set_user_attr(key, value)

        study.optimize(
            objective,
            n_trials=self.optuna_config.get("num_trials"),
            timeout=self.optuna_config.get("timeout_sec", 3600),
            gc_after_trial=True,
        )

    def fit_study(
        self,
        trial: optuna.Trial | None,
        trial_config: ConfigUnion,
        dataset_fold: int,
    ) -> float | None:
        learner, important_config = self.make_learner(
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
            wandb.config.update(important_config)

        trainer = self.make_trainer()
        learner.init()
        trainer.fit(learner)

        if self.use_wandb:
            wandb.finish()

        assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
        best_score = trainer.checkpoint_callback.best_model_score
        return best_score if best_score is None else best_score.item()

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
                    verbose=True,
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

    def wandb_init(self, run_id: str):
        assert "wandb" in self.config
        wandb.init(
            id=run_id,
            tags=self.config["wandb"].get("tags", []),
            project=WANDB_SETTINGS["dummy_project" if self.dummy else "project"],
            group=self.config["learn"]["exp_name"],
            name=self.config["learn"]["run_name"],
            job_type=self.config["wandb"].get("job_type"),
            resume="must" if self.resume else None,
        )
