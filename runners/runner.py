import datetime
import os
import shutil
import time
from abc import ABC, abstractmethod
from typing import Type

import optuna
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
import wandb.errors
import wandb.util
from config.config_maker import make_run_name
from config.config_type import ConfigUnion, LearnerType, RunMode
from config.constants import FILENAMES
from config.optuna import (
    OptunaConfig,
    default_optuna_config,
    pruner_classes,
    sampler_classes,
)
from learners.base_learner import BaseLearner
from learners.typings import BaseLearnerKwargs, DatasetLists
from runners.callbacks import CustomRichProgressBar, custom_rich_progress_bar_theme
from runners.profilers import resolve_profiler
from utils.logging import (
    check_mkdir,
    dump_json,
    get_ckpt_file,
    get_short_git_hash,
    read_from_csv,
)
from utils.optuna import get_optuna_storage, get_study_best_name
from utils.utils import mean
from utils.wandb import (
    get_wandb_project,
    prepare_artifact_name,
    prepare_ckpt_artifact_alias,
    prepare_study_ckpt_artifact_name,
    prepare_study_ref_artifact_name,
    wandb_delete_files,
    wandb_download_ckpt,
    wandb_download_config,
    wandb_get_run_id_by_name,
    wandb_log_file,
    wandb_login,
)


class Runner(ABC):
    def __init__(
        self,
        config: ConfigUnion,
        mode: RunMode,
        learner_type: LearnerType,
        dummy: bool,
        dataset: str = "all",
        resume: bool = False,
    ):
        self.config = config
        self.mode = mode
        self.learner_type = learner_type
        self.dummy = dummy
        self.dataset = dataset
        self.resume = resume

        self.use_wandb = self.config.get("wandb") is not None
        self.exp_name = self.config["learn"]["exp_name"]
        self.run_name = self.config["learn"]["run_name"]
        self.seed = self.config["learn"].get("seed", 0)

        self.optuna_config = self.make_optuna_config()
        self.git_hash = get_short_git_hash()

        self.curr_dataset_fold = -1
        self.curr_trial_number = -1
        self.study_trials = -1
        self.study_end_time = -1

        self.number_of_multi = 0
        self.last_of_multi = False
        self.limit_of_multi = 200

    @abstractmethod
    def make_learner(
        self,
        dataset_fold: int = 0,
        optuna_trial: optuna.Trial | None = None,
    ) -> tuple[Type[BaseLearner], BaseLearnerKwargs]:
        pass

    def update_config(self, optuna_trial: optuna.Trial | None = None) -> dict:
        return {}

    def update_attr(self, exp_name: str | None = None, run_name: str | None = None):
        if exp_name is not None:
            self.config["learn"]["exp_name"] = exp_name
            self.exp_name = exp_name
        if run_name is not None:
            self.config["learn"]["run_name"] = run_name
            self.run_name = run_name

    def make_optuna_config(self) -> OptunaConfig:
        return default_optuna_config

    def make_trainer(self, **kwargs) -> Trainer:
        reload_dataloaders = self.config["data"]["num_workers"] > 0
        callbacks = self.make_callbacks()
        progress = self.config["callbacks"].get("progress", True)
        profiler = resolve_profiler(
            self.config["learn"].get("profiler"),
            os.path.join(self.exp_name, self.run_name),
        )
        return Trainer(
            max_epochs=self.config["learn"]["num_epochs"],
            check_val_every_n_epoch=self.config["learn"].get("val_freq", 1),
            reload_dataloaders_every_n_epochs=1 if reload_dataloaders else 0,
            callbacks=callbacks,
            deterministic=self.config["learn"].get("cudnn_deterministic", "warn"),
            benchmark=self.config["learn"].get("cudnn_benchmark", False),
            logger=False,
            enable_progress_bar=progress,
            enable_model_summary=progress,
            inference_mode=not self.config["learn"].get("manual_optim", False),
            profiler=profiler,
            **kwargs,
        )

    def run_fit_test(
        self,
        fit_only: bool = False,
        test_only: bool = False,
    ):
        seed_everything(self.seed, workers=True)

        important_config = self.update_config()

        if test_only and self.config["learn"].get("ref_ckpt") is None:
            self.resume = True

        if self.use_wandb:
            wandb_login()
            if self.resume:
                run_id = wandb_get_run_id_by_name(self.run_name, dummy=self.dummy)
            else:
                run_id = wandb.util.generate_id()
            assert "wandb" in self.config
            self.config["wandb"]["run_id"] = run_id
            self.wandb_init(run_id, resume=self.resume)
            if self.resume:
                wandb_download_config(self.exp_name, self.run_name)

        ckpt_path = self.resolve_ckpt()

        learner_class, learner_kwargs = self.make_learner()
        if ckpt_path is None:
            learner = learner_class(**learner_kwargs)
        else:
            learner = learner_class.load_from_checkpoint(ckpt_path, **learner_kwargs)

        if self.use_wandb:
            dataset_names = self.get_dataset_names_from_kwargs(learner_kwargs)
            wandb.config.update(
                {"git": self.git_hash, **dataset_names, **important_config}
            )
        init_ok = learner.init(
            resume=self.resume,
            force_clear_dir=True,
            git_hash=self.git_hash,
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
        self.wandb_log_profiles()

        if self.use_wandb:
            wandb.finish()

        self.clean_log_on_end()

    def run_multi_fit_test(self, fit_only: bool = False, test_only: bool = False):
        while self.number_of_multi < self.limit_of_multi and not self.last_of_multi:
            self.update_attr(run_name=make_run_name())
            self.run_fit_test(fit_only, test_only)
            self.number_of_multi += 1
            time.sleep(30)

    def run_study(self):
        def objective(trial: optuna.Trial) -> float:
            info_message = f"Trial {trial.number} started"
            if self.study_trials != -1:
                info_message += f" | {self.study_trials - trial.number } trials left"
            if self.study_end_time != -1:
                time_delta = datetime.timedelta(
                    seconds=self.study_end_time
                    - int(datetime.datetime.now().timestamp())
                )
                info_message += f" | {time_delta} left"
            optuna.logging.get_logger("optuna").log(optuna.logging.INFO, info_message)

            scores = []
            base_run_name = make_run_name()
            self.update_attr(run_name=base_run_name)
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
                self.update_attr(run_name=new_run_name)
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
            study.set_user_attr("git_hash", self.git_hash)
            study.set_user_attr("exp_name", self.exp_name)
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
        if n_trials is not None:
            self.study_trials = n_trials
        if timeout is not None:
            self.study_end_time = int(datetime.datetime.now().timestamp()) + timeout
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
        seed_everything(self.seed, workers=True)

        important_config = self.update_config(trial)

        if self.use_wandb:
            run_id = wandb.util.generate_id()
            assert "wandb" in self.config
            self.config["wandb"]["run_id"] = run_id

        learner_class, learner_kwargs = self.make_learner(
            dataset_fold=self.curr_dataset_fold,
            optuna_trial=trial,
        )
        learner = learner_class(**learner_kwargs)

        study_id = self.optuna_config["study_name"].split(" ")[-1]
        additional_config: dict = {
            "git": self.git_hash,
            "study": study_id,
        }
        dataset_names = self.get_dataset_names_from_kwargs(learner_kwargs)

        if (
            self.use_wandb
            and self.curr_trial_number == 0
            and self.curr_dataset_fold == 0
        ):
            wandb_login()
            wandb.init(
                tags=["helper"],
                project=get_wandb_project(self.dummy),
                group=self.exp_name,
                name=f"log study-ref {study_id}",
                job_type="study",
                settings=wandb.Settings(_disable_stats=True),
            )
            wandb.config.update(additional_config | dataset_names)

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
            additional_config["fold"] = self.curr_dataset_fold
            wandb.config.update(additional_config | dataset_names | important_config)
        learner.init(git_hash=self.git_hash)

        trainer = self.make_trainer()
        trainer.fit(learner)
        self.wandb_log_profiles()

        if self.use_wandb:
            if self.config.get("wandb", {}).get("log_metrics"):
                wandb.log({"pruned": learner.optuna_pruned})
            wandb.finish()

        self.clean_log_on_end()

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

    def get_dataset_names_from_kwargs(
        self, kwargs: BaseLearnerKwargs
    ) -> dict[str, str]:
        dataset_lists: DatasetLists = {
            "dataset_list": kwargs["dataset_list"],
        }
        val_dataset_list = kwargs.get("val_dataset_list")
        if val_dataset_list is not None:
            dataset_lists["val_dataset_list"] = val_dataset_list
        test_dataset_list = kwargs.get("test_dataset_list")
        if test_dataset_list is not None:
            dataset_lists["test_dataset_list"] = test_dataset_list
        return {
            key.replace("_list", ""): ",".join(
                [
                    (kwargs.get("dataset_name") or "NO-NAME")
                    for _, kwargs in dataset_lists[key]
                ]
            )
            for key in dataset_lists
        }

    def wandb_init(self, run_id: str, resume: bool = False):
        assert "wandb" in self.config
        wandb_settings = wandb.Settings(
            _disable_stats=not self.config["wandb"].get("log_system_metrics", False)
        )
        if resume:
            wandb.init(id=run_id, resume="must", settings=wandb_settings)
            return
        wandb.init(
            id=run_id,
            tags=self.config["wandb"].get("tags", []),
            project=get_wandb_project(self.dummy),
            group=self.config["learn"]["exp_name"],
            name=self.config["learn"]["run_name"],
            job_type=self.config["wandb"].get("job_type"),
            settings=wandb_settings,
        )

    def wandb_log_trial(
        self, run_id: str | None, trial: optuna.Trial, new_score: float, pruned: bool
    ):
        if run_id is None:
            return

        self.wandb_init(run_id, resume=True)

        if self.config.get("wandb", {}).get("log_metrics"):
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
            wandb_delete_files(
                artifact_name,
                "study-checkpoint",
                dummy=self.config["learn"].get("dummy") is True,
            )
        except (TypeError, wandb.errors.CommError):
            pass

        index = 0 if self.config["callbacks"].get("monitor_mode") == "min" else -1
        base_run_name = " ".join(self.config["learn"]["run_name"].split(" ")[:3])
        exp_path = os.path.join(FILENAMES["log_folder"], self.exp_name)
        run_names = filter(
            lambda x: x.startswith(base_run_name),
            os.listdir(exp_path),
        )
        for run_name in run_names:
            log_path = os.path.join(exp_path, run_name)
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

    def wandb_log_profiles(self):
        if not self.use_wandb or self.config["learn"].get("profiler") is None:
            return

        log_path = os.path.join(FILENAMES["log_folder"], self.exp_name, self.run_name)
        if not os.path.exists(log_path):
            return
        filenames = filter(lambda x: FILENAMES["profile"] in x, os.listdir(log_path))
        for filename in filenames:
            name, ext = filename.rsplit(".", maxsplit=1)
            filepath = os.path.join(log_path, filename)
            if ext == "csv":
                data = read_from_csv(filepath)
                columns = data.pop(0)
                try:
                    wandb.log({name: wandb.Table(data=data, columns=columns)})
                    continue
                except Exception:
                    pass
            artifact_name = prepare_artifact_name(
                self.config["learn"]["exp_name"], self.config["learn"]["run_name"], name
            )
            wandb_log_file(wandb.run, artifact_name, filepath, "profile")

    def resolve_ckpt(self) -> str | None:
        # ref_ckpt:
        # "{exp}/{run}" | "{exp}/{run}:{direction}" | "{exp}/{run}/{file}.ckpt"
        # "wandb:{ckpt_art}" | "wandb:{ckpt_art}:{direction}" | "wandb:{ckpt_art}:{alias}"
        # "study:{study_id}:{direction}" | "study:{study_id}:{direction}-{fold}"
        # "wandb_study:{study_ckpt_art}" | "wandb_study:{study_ckpt_art}:{direction}" | "wandb_study:{study_ckpt_art}:{alias}"
        # direction: "max" | "min"
        # ckpt_art: "{exp}-{run}-ckpt"
        # study_ckpt_art: "{study_id}-study-ckpt"
        log = FILENAMES["log_folder"]
        if self.resume:
            return os.path.join(log, self.exp_name, self.run_name, "last.ckpt")
        ref_ckpt = self.config["learn"].get("ref_ckpt")
        if ref_ckpt is None:
            return None
        if ref_ckpt.startswith("wandb"):
            splitted = ref_ckpt.split(":")
            ckpt_art = splitted[1]
            alias = splitted[2] if len(splitted) == 3 else None
            if ref_ckpt.startswith("wandb_study"):
                study = True
                study_id = ckpt_art.split("-")[0]
                run_name, exp_name = get_study_best_name(study_id, self.dummy)
                exp_name = exp_name or self.exp_name
                run_name = run_name or f"best {study_id}"
            else:
                study = False
                exp_name, run_date, run_time, run_id, _ = ckpt_art.rsplit(
                    "-", maxsplit=4
                )
                run_date = run_date[:4] + "-" + run_date[4:6] + "-" + run_date[6:]
                run_time = run_time[:2] + "-" + run_time[2:]
                run_name = f"{run_date} {run_time} {run_id}"
            return wandb_download_ckpt(
                ckpt_art,
                os.path.join(log, exp_name, run_name),
                alias,
                dummy=self.dummy,
                study=study,
            )
        if ref_ckpt.startswith("study"):
            exp_path = os.path.join(log, self.exp_name)
            _, study_id, direction = ref_ckpt.split(":")
            if "-" in direction:
                direction, fold = direction.split("-")
            else:
                fold = None
            index = -1 if direction == "max" else 0
            base_run_name, exp_name = get_study_best_name(study_id, self.dummy)
            if base_run_name is None:
                raise ValueError(f"Name of the best run for study {study_id} not found")
            exp_name = exp_name or self.exp_name
            if fold is not None:
                run_name = f"{base_run_name} F{int(fold)}"
                ckpt = get_ckpt_file(self.exp_name, run_name, index)
                return os.path.join(exp_path, run_name, ckpt)
            run_names = sorted(
                filter(lambda x: x.startswith(base_run_name), os.listdir(exp_path))
            )
            run_ckpts = [
                (rn, get_ckpt_file(self.exp_name, rn, index)) for rn in run_names
            ]
            reduce_func = max if direction == "max" else min
            run_name, ckpt = reduce_func(run_ckpts, key=lambda x: x[1])
            return os.path.join(exp_path, run_name, ckpt)
        splitted = ref_ckpt.split("/")
        if len(splitted) == 3:
            return os.path.join(log, *splitted)
        exp_name, run_name = splitted
        if ":" in run_name:
            run_name, direction = run_name.split(":")
            index = -1 if direction == "max" else 0
            ckpt = get_ckpt_file(exp_name, run_name, index)
        else:
            ckpt = "last.ckpt"
        return os.path.join(log, exp_name, run_name, ckpt)

    def clean_log_on_end(self):
        if not self.config["log"].get("clean_on_end", False):
            return
        log_path = os.path.join(FILENAMES["log_folder"], self.exp_name, self.run_name)
        shutil.rmtree(log_path, ignore_errors=True)
