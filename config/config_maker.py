import datetime
import os
from typing import Literal

from config.config_type import (
    CallbacksConfig,
    ConfigBase,
    ConfigGuidedNets,
    ConfigMetaLearner,
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigUnion,
    ConfigWeasel,
    DataConfig,
    GuidedNetsConfig,
    LearnConfig,
    LossConfig,
    MetaLearnerConfig,
    OptimizerConfig,
    ProtoSegConfig,
    SchedulerConfig,
    SimpleLearnerConfig,
    WandbConfig,
    WeaselConfig,
)
from config.constants import FILENAMES
from utils.utils import generate_char

data_config: DataConfig = {
    "num_classes": 3,
    "num_channels": 3,
    "num_workers": 0,
    "batch_size": 2,
    "resize_to": (256, 256),
}

learn_config: LearnConfig = {
    "num_epochs": 5,
    "exp_name": "dummy",
    "run_name": "",
    "dummy": True,
    "val_freq": 1,
    "tensorboard_graph": True,
    "ref_ckpt_path": None,
}

loss_config: LossConfig = {"type": "ce", "ignored_index": -1}

optimizer_config: OptimizerConfig = {
    "lr": 1e-3,
    "lr_bias": 2e-3,
    "weight_decay": 5e-5,
    "weight_decay_bias": 0,
    "betas": (0.9, 0.99),
}

scheduler_config: SchedulerConfig = {"step_size": 150, "gamma": 0.2}

callbacks_config: CallbacksConfig = {
    "progress_leave": True,
    "monitor": "val_score",
    "monitor_mode": "max",
    "ckpt_top_k": 5,
    "stop_patience": 5,
    "stop_min_delta": 0.0,
    "stop_threshold": None,
}

wandb_config: WandbConfig = {
    "run_id": "",
    "tags": [],
    "job_type": None,
    "log_model": True,
    "watch_model": True,
    "push_table_freq": 5,
    "sweep_metric": ("summary/val_score", "maximize"),
    "sweep_id": "",
    "save_train_preds": 0,
    "save_val_preds": 0,
    "save_test_preds": 0,
}

config_base: ConfigBase = {
    "data": data_config,
    "learn": learn_config,
    "loss": loss_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
    "callbacks": callbacks_config,
    "wandb": wandb_config,
}

simple_learner_config: SimpleLearnerConfig = {}

meta_learner_config: MetaLearnerConfig = {}

weasel_config: WeaselConfig = {
    "use_first_order": False,
    "update_param_step_size": 0.3,
    "tune_epochs": 10,
    "tune_val_freq": 1,
}

protoseg_config: ProtoSegConfig = {
    "embedding_size": 8,
}

guidednets_config: GuidedNetsConfig = {"embedding_size": 32}


def make_run_name(exp_name: str) -> str:
    run_name_ori = (
        datetime.datetime.now().isoformat()[0:16].replace(":", "-").replace("T", " ")
    )
    exp_path = os.path.join(FILENAMES["log_folder"], exp_name)
    if not os.path.exists(exp_path):
        return run_name_ori
    existing_runs = os.listdir(exp_path)

    i = 0
    run_name = run_name_ori
    while run_name in existing_runs:
        run_name = run_name_ori + " " + generate_char(i)

    return run_name


def make_config(
    learner: Literal["simple", "meta", "weasel", "protoseg", "guidednets", None] = None,
    mode: Literal["fit", "test", "sweep", None] = None,
    name_suffix: str = "",
    use_wandb: bool = True,
    dummy: bool = False,
) -> ConfigUnion:
    if mode == "sweep":
        use_wandb = True
        config_base["learn"]["tensorboard_graph"] = False
        config_base["wandb"] = {
            "run_id": "",
            "tags": [],
            "job_type": "sweep",
            "log_model": False,
            "watch_model": False,
            "push_table_freq": None,
            "sweep_metric": ("summary/val_score", "maximize"),
            "save_train_preds": 0,
            "save_val_preds": 0,
            "save_test_preds": 0,
        }
    elif mode == "test":
        config_base["learn"]["tensorboard_graph"] = False
        if use_wandb:
            config_base["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": "test",
                "log_model": False,
                "watch_model": False,
                "push_table_freq": 1,
                "sweep_metric": None,
                "save_train_preds": 0,
                "save_val_preds": 0,
                "save_test_preds": 20,
            }
    elif mode == "fit":
        config_base["learn"]["tensorboard_graph"] = True
        if use_wandb:
            config_base["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": "fit",
                "log_model": True,
                "watch_model": True,
                "push_table_freq": 5,
                "sweep_metric": None,
                "save_train_preds": 20,
                "save_val_preds": 20,
                "save_test_preds": 20,
            }

    save_train_preds = config_base.get("wandb", {}).get("save_train_preds", 0)
    save_val_preds = config_base.get("wandb", {}).get("save_val_preds", 0)
    save_test_preds = config_base.get("wandb", {}).get("save_test_preds", 0)
    if dummy:
        config_base["learn"]["dummy"] = True
        config_base["data"]["num_workers"] = 0
        config_base["callbacks"]["ckpt_top_k"] = 3
        save_train_preds //= 5
        save_val_preds //= 5
        save_test_preds //= 5
    else:
        config_base["learn"]["dummy"] = False
        config_base["data"]["num_workers"] = 3
    if use_wandb:
        config_base["wandb"] = {  # type: ignore
            **config_base.get("wandb", {}),
            **{
                "save_train_preds": save_train_preds,
                "save_val_preds": save_val_preds,
                "save_test_preds": save_test_preds,
            },
        }
    else:
        config_base.pop("wandb")

    config: ConfigUnion = config_base.copy()
    if learner == "simple":
        config_simple: ConfigSimpleLearner = {
            **config_base,
            "simple_learner": simple_learner_config,
        }
        config_simple["learn"]["exp_name"] = "SL"
        if not dummy:
            config_simple["data"]["batch_size"] = 32
            config_simple["learn"]["num_epochs"] = 300
        config = config_simple
    elif learner == "meta":
        config_meta: ConfigMetaLearner = {
            **config_base,
            "meta_learner": meta_learner_config,
        }
        config_meta["learn"]["exp_name"] = "ML"
        if not dummy:
            config_meta["data"]["batch_size"] = 13
            config_meta["learn"]["num_epochs"] = 100
        config = config_meta
    elif learner == "weasel":
        config_weasel: ConfigWeasel = {
            **config_base,
            "meta_learner": meta_learner_config,
            "weasel": weasel_config,
        }
        config_weasel["learn"]["exp_name"] = "WS"
        if not dummy:
            config_weasel["data"]["batch_size"] = 13
            config_weasel["learn"]["num_epochs"] = 100
        config = config_weasel
    elif learner == "protoseg":
        config_protoseg: ConfigProtoSeg = {
            **config_base,
            "meta_learner": meta_learner_config,
            "protoseg": protoseg_config,
        }
        config_protoseg["learn"]["exp_name"] = "PS"
        if not dummy:
            config_protoseg["data"]["batch_size"] = 32
            config_protoseg["learn"]["num_epochs"] = 100
        config = config_protoseg
    elif learner == "guidednets":
        config_guidednets: ConfigGuidedNets = {
            **config_base,
            "meta_learner": meta_learner_config,
            "guidednets": guidednets_config,
        }
        config_guidednets["learn"]["exp_name"] = "GN"
        if not dummy:
            config_guidednets["data"]["batch_size"] = 8
            config_guidednets["learn"]["num_epochs"] = 100
        config = config_guidednets

    exp_name = config["learn"]["exp_name"]
    exp_name += " " + name_suffix
    exp_name = exp_name.strip()
    config["learn"]["exp_name"] = exp_name
    config["learn"]["run_name"] = make_run_name(exp_name)

    return config
