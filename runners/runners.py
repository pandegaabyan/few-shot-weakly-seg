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
        resume: bool = ...,
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

    learner, trainer = make_learner_and_trainer(
        config, dummy, resume=resume, learner_ckpt=ckpt_path
    )
    if not learner.init():
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


# def run_sweep(config: ConfigUnion, dummy: bool, use_cv: bool = False, count: int = 3):
#     assert "wandb" in config

#     sweep_config: SweepConfigBase = {
#         "method": "random",
#         "metric": {"name": "summary/val_score", "goal": "maximize"},
#         "parameters": {
#             "opt_lr": {
#                 "distribution": "log_uniform_values",
#                 "min": 0.00001,
#                 "max": 0.1,
#             },
#             "opt_weight_decay": {
#                 "distribution": "log_uniform_values",
#                 "min": 0.00001,
#                 "max": 0.1,
#             },
#             "opt_beta_0": {"values": [0.5, 0.9, 0.99]},
#             "opt_beta_1": {"values": [0.99, 0.999, 0.9999]},
#             "sch_gamma": {"distribution": "uniform", "min": 0.05, "max": 0.95},
#             # "dataset_fold": {"values": [0, 1, 2, 3]},
#         },
#     }

#     def update_config_from_ref(config: ConfigUnion, ref_config: dict):
#         config["optimizer"]["lr"] = ref_config["opt_lr"]
#         config["optimizer"]["lr_bias"] = ref_config["opt_lr"] * 2
#         config["optimizer"]["weight_decay"] = ref_config["opt_weight_decay"]
#         config["optimizer"]["betas"] = (
#             ref_config["opt_beta_0"],
#             ref_config["opt_beta_1"],
#         )
#         config["scheduler"]["gamma"] = ref_config["sch_gamma"]

#     ref_ckpt_path = config["learn"].get("ref_ckpt_path")
#     ckpt_path = ref_ckpt_path and get_full_ckpt_path(ref_ckpt_path)

#     sweep_config = initialize_sweep(config, sweep_config, dummy, use_cv)
#     config["wandb"]["sweep_id"] = sweep_config["sweep_id"]

#     def train(
#         config: ConfigUnion = deepcopy(config), ref_config: dict | None = None
#     ) -> float | None:
#         assert "wandb" in config

#         if use_cv:
#             sweep_parent = config["wandb"].get("sweep_parent")
#             assert sweep_parent and ref_config
#             dataset_fold = ref_config["dataset_fold"]
#             run_name = f"{sweep_parent} F{dataset_fold}"
#             project = WANDB_SETTINGS["dummy_project" if dummy else "project"]
#             update_config_from_ref(config, ref_config)
#         else:
#             run_name = make_run_name(config["learn"]["exp_name"])
#             project = None
#         config["learn"]["run_name"] = run_name

#         wandb.init(
#             config=dict(config) if use_cv else None,
#             tags=config["wandb"]["tags"],
#             project=project,
#             group=config["learn"]["exp_name"],
#             name=run_name,
#             job_type=config["wandb"]["job_type"],
#             reinit=True if use_cv else None,
#         )
#         assert wandb.run
#         config["wandb"]["run_id"] = wandb.run.id

#         if not use_cv:
#             ref_config = dict(wandb.config)
#             dataset_fold = ref_config.get("dataset_fold", 0)
#             update_config_from_ref(config, ref_config)

#         learner, trainer = make_learner_and_trainer(
#             config, dummy, dataset_fold=dataset_fold, learner_ckpt=ckpt_path
#         )
#         if not learner.init():
#             return None

#         trainer.fit(learner)
#         monitor = config["callbacks"].get("monitor")
#         final_score = trainer.callback_metrics[monitor].item() if monitor else None

#         wandb.finish()
#         return final_score

#     def train_cv(config: ConfigUnion = deepcopy(config)):
#         num_folds = 4

#         assert "wandb" in config

#         config["learn"]["run_name"] = make_run_name(config["learn"]["exp_name"])

#         wandb.init(
#             tags=config["wandb"]["tags"] + ["sweep-parent"],
#             group=config["learn"]["exp_name"],
#             name=config["learn"]["run_name"],
#             job_type=config["wandb"]["job_type"],
#         )
#         sweep_run_id = wandb.run and wandb.run.id
#         ref_config = dict(wandb.config)
#         wandb.finish()
#         wandb_setup._setup(_reset=True)

#         config["wandb"]["sweep_parent"] = config["learn"]["run_name"]

#         scores = []
#         for i in range(num_folds):
#             reset_wandb_env()
#             ref_config["dataset_fold"] = i
#             final_score = train(config, ref_config)
#             if final_score is not None:
#                 scores.append(final_score)

#         wandb.init(id=sweep_run_id, resume="must")
#         wandb.log({sweep_config["metric"]["name"]: mean(scores)})
#         wandb.finish()

#     wandb.agent(
#         sweep_config["sweep_id"],
#         function=train_cv if use_cv else train,
#         project=WANDB_SETTINGS["dummy_project" if dummy else "project"],
#         count=count,
#     )
