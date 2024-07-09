from typing import Protocol

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import wandb
from config.config_type import ConfigUnion
from config.constants import WANDB_SETTINGS
from learners.base_learner import BaseLearner
from runners.trainer import make_trainer
from utils.logging import (
    get_configuration,
    get_full_ckpt_path,
)
from utils.wandb import (
    wandb_login,
)
from wandb import util as wandb_util


class MakeLearnerTrainer(Protocol):
    def __call__(
        self,
        config: ConfigUnion,
        dummy: bool,
        dataset_fold: int = ...,
        learner_ckpt: str | None = ...,
    ) -> tuple[BaseLearner, Trainer]:
        ...


def run_fit_test(
    config: ConfigUnion,
    dummy: bool,
    make_learner_and_trainer: MakeLearnerTrainer,
    resume: bool = False,
    fit_only: bool = False,
    test_only: bool = False,
):
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
            run_id = wandb_util.generate_id()
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

    learner, trainer = make_learner_and_trainer(config, dummy, learner_ckpt=ckpt_path)
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
        wandb.finish()


def run_optimization(
    config: ConfigUnion,
    dummy: bool,
    use_cv: bool = False,
    count: int | None = None,
    duration: int | None = None,
):
    ...
