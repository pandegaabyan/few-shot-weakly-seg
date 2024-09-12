import datetime
from copy import deepcopy

import nanoid

from config.config_type import (
    CallbacksConfig,
    ConfigBase,
    ConfigGuidedNets,
    ConfigProtoSeg,
    ConfigSimpleLearner,
    ConfigUnion,
    ConfigWeasel,
    DataConfig,
    GuidedNetsConfig,
    LearnConfig,
    LearnerType,
    LogConfig,
    MetaLearnerConfig,
    OptimizerConfig,
    ProtoSegConfig,
    RunMode,
    SchedulerConfig,
    SimpleLearnerConfig,
    WandbConfig,
    WeaselConfig,
)

data_config: DataConfig = {
    "num_classes": 3,
    "num_channels": 3,
    "num_workers": 0,
    "batch_size": 2,
    "resize_to": (256, 256),
}

learn_config: LearnConfig = {
    "num_epochs": 4,
    "exp_name": "dummy",
    "run_name": "",
    "dummy": True,
    "val_freq": 1,
    "cudnn_deterministic": "warn",
    "cudnn_benchmark": False,
    "profiler": None,
    "manual_optim": False,
    "ref_ckpt": None,
    "optuna_study": None,
}

optimizer_config: OptimizerConfig = {
    "lr": 1e-3,
    "lr_bias_mult": 1,
    "weight_decay": 5e-5,
    "betas": (0.9, 0.99),
}

scheduler_config: SchedulerConfig = {"step_size": 10, "gamma": 0.1}

log_config: LogConfig = {
    "configuration": True,
    "table": True,
    # "model_onnx": True,
    # "tensorboard_graph": True,
    "model_onnx": False,
    "tensorboard_graph": False,
}

callbacks_config: CallbacksConfig = {
    "progress": True,
    "monitor": "val_score",
    "monitor_mode": "max",
    "ckpt_last": True,
    "ckpt_top_k": 2,
    "stop_patience": 2,
    "stop_min_delta": 0.0,
    "stop_threshold": None,
}

wandb_config: WandbConfig = {
    "run_id": "",
    "tags": [],
    "job_type": None,
    "watch_model": True,
    "save_model": True,
    "push_table_freq": 5,
    "save_mask_only": False,
    "save_train_preds": 0,
    "save_val_preds": 0,
    "save_test_preds": 0,
}

config_base: ConfigBase = {
    "data": data_config,
    "learn": learn_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
    "log": log_config,
    "callbacks": callbacks_config,
}

simple_learner_config: SimpleLearnerConfig = {}

meta_learner_config: MetaLearnerConfig = {}

weasel_config: WeaselConfig = {
    "first_order": False,
    "update_param_rate": 0.3,
    "tune_epochs": 3,
    "tune_val_freq": None,
    "tune_multi_step": False,
}

protoseg_config: ProtoSegConfig = {
    "multi_pred": False,
    "embedding_size": 4,
}

guidednets_config: GuidedNetsConfig = {"embedding_size": 4}


def gen_id(size: int) -> str:
    return nanoid.generate(
        size=size,
        alphabet="_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )


def make_run_name() -> str:
    timestamp_text = (
        datetime.datetime.now().isoformat()[0:16].replace(":", "-").replace("T", " ")
    )
    random_id = gen_id(3)
    return f"{timestamp_text} {random_id}"


def make_config(
    learner: LearnerType = None,
    mode: RunMode = "fit-test",
    name_suffix: str = "",
    use_wandb: bool = True,
    dummy: bool = False,
) -> ConfigUnion:
    config_ref = deepcopy(config_base)

    if mode in ["profile-fit", "profile-test"]:
        config_ref["learn"]["cudnn_deterministic"] = False
        config_ref["learn"]["cudnn_benchmark"] = False
        config_ref["learn"]["profiler"] = "custom-1"
        config_ref["log"]["configuration"] = False
        config_ref["log"]["table"] = False
        config_ref["log"]["model_onnx"] = False
        config_ref["log"]["tensorboard_graph"] = False
        config_ref["callbacks"]["progress"] = False
        config_ref["callbacks"]["ckpt_last"] = False
        config_ref["callbacks"]["ckpt_top_k"] = 0
        config_ref["callbacks"]["stop_patience"] = 500
        if use_wandb:
            config_ref["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": mode,
                "watch_model": False,
                "save_model": False,
                "push_table_freq": None,
                "save_mask_only": False,
                "save_train_preds": 0,
                "save_val_preds": 0,
                "save_test_preds": 0,
            }
    elif mode == "study":
        config_ref["learn"]["cudnn_deterministic"] = False
        config_ref["learn"]["cudnn_benchmark"] = True
        config_ref["log"]["configuration"] = False
        config_ref["log"]["table"] = False
        config_ref["log"]["model_onnx"] = False
        config_ref["log"]["tensorboard_graph"] = False
        config_ref["callbacks"]["progress"] = False
        config_ref["callbacks"]["ckpt_last"] = False
        config_ref["callbacks"]["ckpt_top_k"] = 1
        if use_wandb:
            config_ref["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": mode,
                "watch_model": False,
                "save_model": False,
                "push_table_freq": 20,
                "save_mask_only": False,
                "save_train_preds": 0,
                "save_val_preds": 0,
                "save_test_preds": 0,
            }
    elif mode == "fit":
        if use_wandb:
            config_ref["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": mode,
                "watch_model": True,
                "save_model": True,
                "push_table_freq": 5,
                "save_mask_only": False,
                "save_train_preds": 40,
                "save_val_preds": 40,
                "save_test_preds": 0,
            }
    elif mode == "test":
        config_ref["log"]["model_onnx"] = False
        config_ref["log"]["tensorboard_graph"] = False
        if use_wandb:
            config_ref["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": mode,
                "watch_model": False,
                "save_model": False,
                "push_table_freq": 1,
                "save_mask_only": False,
                "save_train_preds": 0,
                "save_val_preds": 0,
                "save_test_preds": 40,
            }
    else:
        if use_wandb:
            config_ref["wandb"] = {
                "run_id": "",
                "tags": [],
                "job_type": mode,
                "watch_model": True,
                "save_model": True,
                "push_table_freq": 5,
                "save_mask_only": False,
                "save_train_preds": 40,
                "save_val_preds": 40,
                "save_test_preds": 40,
            }

    if not dummy:
        config_ref["data"]["num_workers"] = 3
        config_ref["learn"]["dummy"] = False
        if mode != "study":
            config_ref["callbacks"]["ckpt_top_k"] = 5

    if dummy and use_wandb:
        assert "wandb" in config_ref
        for key in ["save_train_preds", "save_val_preds", "save_test_preds"]:
            if key in config_ref["wandb"]:
                config_ref["wandb"][key] //= 5

    config: ConfigUnion = deepcopy(config_ref)
    if learner == "simple":
        config_simple: ConfigSimpleLearner = {
            **config_ref,
            "simple_learner": simple_learner_config,
        }
        config_simple["learn"]["exp_name"] = "SL"
        if not dummy:
            config_simple["data"]["batch_size"] = 16
            config_simple["learn"]["num_epochs"] = 200
            config_simple["learn"]["val_freq"] = 1
            config_simple["callbacks"]["stop_patience"] = 30
        config = config_simple
    elif learner == "weasel":
        config_weasel: ConfigWeasel = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "weasel": weasel_config,
        }
        config_weasel["learn"].update({"exp_name": "WS", "manual_optim": True})
        if not dummy:
            config_weasel["data"]["batch_size"] = 8
            config_weasel["learn"]["num_epochs"] = 100
            config_weasel["learn"]["val_freq"] = 10
            config_weasel["callbacks"]["stop_patience"] = 2
            config_weasel["weasel"]["tune_epochs"] = 20
        else:
            config_weasel["data"]["batch_size"] = 1
            config_weasel["learn"]["num_epochs"] = 2
        config = config_weasel
    elif learner == "protoseg":
        config_protoseg: ConfigProtoSeg = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "protoseg": protoseg_config,
        }
        config_protoseg["learn"]["exp_name"] = "PS"
        if not dummy:
            config_protoseg["data"]["batch_size"] = 12
            config_protoseg["learn"]["num_epochs"] = 100
            config_protoseg["learn"]["val_freq"] = 2
            config_protoseg["callbacks"]["stop_patience"] = 10
        else:
            config_protoseg["data"]["batch_size"] = 1
            config_protoseg["learn"]["num_epochs"] = 2
        config = config_protoseg
    elif learner == "guidednets":
        config_guidednets: ConfigGuidedNets = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "guidednets": guidednets_config,
        }
        config_guidednets["learn"]["exp_name"] = "GN"
        if not dummy:
            config_guidednets["data"]["batch_size"] = 8
            config_guidednets["learn"]["num_epochs"] = 100
            config_guidednets["learn"]["val_freq"] = 2
            config_guidednets["callbacks"]["stop_patience"] = 10
        else:
            config_guidednets["data"]["batch_size"] = 1
            config_guidednets["learn"]["num_epochs"] = 2

        config = config_guidednets

    exp_name = config["learn"]["exp_name"]
    exp_name += " " + name_suffix
    exp_name = exp_name.strip()
    config["learn"]["exp_name"] = exp_name
    if mode != "study":
        config["learn"]["run_name"] = make_run_name()

    return config
