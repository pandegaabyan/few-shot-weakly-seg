import datetime
from copy import deepcopy

import nanoid

from config.config_type import (
    CallbacksConfig,
    ConfigBase,
    ConfigPANet,
    ConfigPASNet,
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
    ModelConfig,
    OptimizerConfig,
    PANetConfig,
    PASNetConfig,
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
    "exp_name": "dummy",
    "run_name": "",
    "num_epochs": 4,
    "dummy": True,
    "seed": 0,
    "val_freq": 1,
    "cudnn_deterministic": "warn",
    "cudnn_benchmark": False,
    "profiler": None,
    "profile_id": None,
    "manual_optim": False,
    "ref_ckpt": None,
    "optuna_study": None,
}

model_config: ModelConfig = {"arch": "unetmini"}

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
    "clean_on_end": False,
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
    "log_metrics": True,
    "log_system_metrics": False,
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
    "model": model_config,
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

panet_config: PANetConfig = {
    "embedding_size": 4,
    "par_weight": 0.5,
    "metric_func": "euclidean",
}

pasnet_config: PASNetConfig = {
    "embedding_size": 4,
    "par_weight": 0.5,
    "consistency_weight": 0.5,
    "prototype_metric_func": "cosine",
    "consistency_metric_func": "euclidean",
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
    learner: LearnerType = "SL",
    mode: RunMode = "fit-test",
    name_suffix: str = "",
    use_wandb: bool = True,
    dummy: bool = False,
) -> ConfigUnion:
    config_ref = deepcopy(config_base)

    if mode in ["profile-fit", "profile-test"]:
        config_ref["learn"]["cudnn_deterministic"] = "warn"
        config_ref["learn"]["cudnn_benchmark"] = False
        config_ref["learn"]["profiler"] = "custom-1"
        config_ref["learn"]["profile_id"] = gen_id(5)
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
                "log_metrics": False,
                "log_system_metrics": True,
                "watch_model": False,
                "save_model": False,
                "push_table_freq": None,
                "save_mask_only": False,
                "save_train_preds": 0,
                "save_val_preds": 0,
                "save_test_preds": 0,
            }
    elif mode == "study":
        config_ref["learn"]["cudnn_deterministic"] = "warn"
        config_ref["learn"]["cudnn_benchmark"] = False
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
                "log_metrics": True,
                "log_system_metrics": False,
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
                "log_metrics": True,
                "log_system_metrics": False,
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
                "log_metrics": True,
                "log_system_metrics": False,
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
                "log_metrics": True,
                "log_system_metrics": False,
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
        if mode in ["fit", "test", "fit-test"]:
            config_ref["callbacks"]["ckpt_top_k"] = 5

    if dummy and use_wandb:
        assert "wandb" in config_ref
        for key in ["save_train_preds", "save_val_preds", "save_test_preds"]:
            if key in config_ref["wandb"]:
                config_ref["wandb"][key] //= 5

    config: ConfigUnion = deepcopy(config_ref)
    runner_name = learner.split("-")[0]
    if runner_name == "SL":
        config_simple: ConfigSimpleLearner = {
            **config_ref,
            "simple_learner": simple_learner_config,
        }
        if not dummy:
            config_simple["data"]["batch_size"] = 16
            config_simple["learn"]["num_epochs"] = 200
            config_simple["learn"]["val_freq"] = 1
            config_simple["callbacks"]["stop_patience"] = 30
        config = config_simple
    elif runner_name == "WS":
        config_weasel: ConfigWeasel = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "weasel": weasel_config,
        }
        config_weasel["learn"].update({"manual_optim": True})
        if not dummy:
            config_weasel["data"]["batch_size"] = 8
            config_weasel["learn"]["num_epochs"] = 100
            config_weasel["learn"]["val_freq"] = 10
            config_weasel["callbacks"]["stop_patience"] = 2
            config_weasel["weasel"]["tune_epochs"] = 20
        else:
            config_weasel["data"]["batch_size"] = 1
            config_weasel["learn"]["num_epochs"] = 2
        if "fo" in learner.split("-"):
            config_weasel["weasel"]["first_order"] = True
        if "ms" in learner.split("-"):
            config_weasel["weasel"]["tune_multi_step"] = True
        if "ori" in learner.split("-"):
            config_weasel["weasel"]["tune_multi_step"] = True
            config_weasel["learn"]["num_epochs"] = 200
            config_weasel["learn"]["val_freq"] = 20
            config_weasel["scheduler"]["step_size"] = 40
            config_weasel["callbacks"]["stop_patience"] = 2
        config = config_weasel
    elif runner_name == "PS":
        config_protoseg: ConfigProtoSeg = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "protoseg": protoseg_config,
        }
        if not dummy:
            config_protoseg["data"]["batch_size"] = 12
            config_protoseg["learn"]["num_epochs"] = 100
            config_protoseg["learn"]["val_freq"] = 2
            config_protoseg["callbacks"]["stop_patience"] = 10
        else:
            config_protoseg["data"]["batch_size"] = 1
            config_protoseg["learn"]["num_epochs"] = 2
        if "mp" in learner.split("-"):
            config_protoseg["protoseg"]["multi_pred"] = True
        if "ori" in learner.split("-"):
            config_protoseg["protoseg"]["multi_pred"] = True
            config_protoseg["protoseg"]["embedding_size"] = 3
            config_protoseg["learn"]["num_epochs"] = 200
            config_protoseg["learn"]["val_freq"] = 4
            config_protoseg["scheduler"]["step_size"] = 40
            config_protoseg["callbacks"]["stop_patience"] = 10
        config = config_protoseg
    elif runner_name == "PA":
        config_panet: ConfigPANet = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "panet": panet_config,
        }
        if not dummy:
            config_panet["data"]["batch_size"] = 12
            config_panet["learn"]["num_epochs"] = 100
            config_panet["learn"]["val_freq"] = 2
            config_panet["callbacks"]["stop_patience"] = 10
        else:
            config_panet["data"]["batch_size"] = 1
            config_panet["learn"]["num_epochs"] = 2
        config = config_panet
    elif runner_name == "PAS":
        config_pasnet: ConfigPASNet = {
            **config_ref,
            "meta_learner": meta_learner_config,
            "pasnet": pasnet_config,
        }
        if not dummy:
            config_pasnet["data"]["batch_size"] = 12
            config_pasnet["learn"]["num_epochs"] = 100
            config_pasnet["learn"]["val_freq"] = 2
            config_pasnet["callbacks"]["stop_patience"] = 10
        else:
            config_pasnet["data"]["batch_size"] = 1
            config_pasnet["learn"]["num_epochs"] = 2
        config = config_pasnet

    exp_name = learner
    exp_name += " " + name_suffix
    exp_name = exp_name.strip()
    config["learn"]["exp_name"] = exp_name
    if mode in ["fit", "test", "fit-test"]:
        config["learn"]["run_name"] = make_run_name()

    return config
