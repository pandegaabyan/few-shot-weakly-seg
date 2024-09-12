import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Generic, Literal, Mapping, Sequence, Type

import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.profilers.profiler import Profiler
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Unpack

import wandb
from config.config_type import ConfigUnion
from config.constants import FILENAMES
from data.base_dataset import BaseDataset
from data.typings import BaseDataTuple, DatasetModes
from learners.losses import CustomLoss
from learners.metrics import MultiIoUMetric
from learners.optimizers import get_optimizer_and_scheduler_names
from learners.typings import (
    BaseLearnerKwargs,
    ConfigType,
    DatasetClass,
    DatasetKwargs,
    ListPrimitives,
    PredictionDataDict,
    Primitives,
    Scheduler,
)
from runners.callbacks import CustomRichProgressBar, ProgressBarTaskType
from utils.diff_dict import diff_dict
from utils.logging import (
    check_mkdir,
    check_rmtree,
    dump_json,
    get_name_from_class,
    get_name_from_instance,
    get_short_git_hash,
    load_json,
    write_to_csv,
)
from utils.time import get_iso_timestamp_now
from utils.utils import (
    mean,
    merge_dicts,
)
from utils.wandb import (
    prepare_artifact_name,
    prepare_ckpt_artifact_alias,
    prepare_ckpt_artifact_name,
    prepare_study_ref_artifact_name,
    wandb_delete_files,
    wandb_log_file,
)


class BaseLearner(
    LightningModule, ABC, Generic[ConfigType, DatasetClass, DatasetKwargs]
):
    def __init__(
        self,
        **kwargs: Unpack[BaseLearnerKwargs[ConfigType, DatasetClass, DatasetKwargs]],
    ):
        super().__init__()

        self.config = kwargs["config"]
        self.dataset_list = kwargs["dataset_list"]
        self.val_dataset_list = kwargs.get("val_dataset_list")
        self.test_dataset_list = kwargs.get("test_dataset_list")
        self.optuna_trial = kwargs.get("optuna_trial")

        self.train_datasets = self.make_dataset("train", self.dataset_list)
        self.val_datasets = self.make_dataset(
            "val", self.val_dataset_list or self.dataset_list
        )
        self.test_datasets = self.make_dataset(
            "test", self.test_dataset_list or self.val_dataset_list or self.dataset_list
        )

        (loss_class, loss_kwargs) = kwargs.get("loss") or (CustomLoss, {})
        self.loss = loss_class(**loss_kwargs)
        self.loss_kwargs = loss_kwargs
        (metric_class, metric_kwargs) = kwargs.get("metric") or (MultiIoUMetric, {})
        self.metric = metric_class(**metric_kwargs)
        self.metric_kwargs = metric_kwargs

        if (
            any(lv for lv in self.config["log"].values())
            or self.config["callbacks"].get("ckpt_last")
            or self.config["callbacks"].get("ckpt_top_k")
            or self.config["learn"].get("profiler")
        ):
            self.log_path = os.path.join(
                FILENAMES["log_folder"],
                self.config["learn"]["exp_name"],
                self.config["learn"]["run_name"],
            )
        else:
            self.log_path = ""

        wandb_config = self.config.get("wandb", {})
        self.train_indices_to_save = self.make_indices_to_save(
            self.train_datasets, wandb_config.get("save_train_preds", 0)
        )
        self.val_indices_to_save = self.make_indices_to_save(
            self.val_datasets, wandb_config.get("save_val_preds", 0)
        )
        self.test_indices_to_save = self.make_indices_to_save(
            self.test_datasets, wandb_config.get("save_test_preds", 0)
        )
        self.class_labels = merge_dicts(
            [
                ds.class_labels
                for ds in (self.train_datasets + self.val_datasets + self.test_datasets)
            ]
        )

        self.use_wandb = self.config.get("wandb") is not None
        self.example_input_array = self.make_input_example()

        self.init_ok = False
        self.resume = False
        self.configuration_logged = False
        self.optuna_pruned = False
        self.best_monitor_value: float | None = None
        self.wandb_tables: dict[str, wandb.Table] = {}
        self.training_step_losses: list[float] = []
        self.validation_step_losses: list[float] = []
        self.test_step_losses: list[float] = []
        self.last_prediction_data: PredictionDataDict = {}
        self.best_prediction_data: PredictionDataDict = {}

    @abstractmethod
    def make_dataloader(self, datasets: list[DatasetClass]) -> DataLoader:
        pass

    @abstractmethod
    def make_indices_to_save(
        self, datasets: list[DatasetClass], sample_size: int
    ) -> list[list[int]] | None:
        pass

    @abstractmethod
    def make_input_example(self) -> tuple[Any, ...]:
        pass

    def on_fit_start(self):
        if not self.init_ok:
            raise ValueError("Learner not initialized")

        super().on_fit_start()

        self.cast_example_input_array()
        self.prepare_datasets()

        self.log_configuration()
        self.log_model_onnx()
        self.log_tensorboard_graph()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        val_loss = mean(self.validation_step_losses)
        self.validation_step_losses.clear()
        val_score = self.metric.compute()
        val_score = self.metric.prepare_for_log(val_score)
        score_summary = self.metric.score_summary()

        if self.trainer.sanity_checking:
            return

        self.wandb_log(
            {"loss": val_loss} | dict(val_score) | {"score": score_summary},
            "summary/val_",
        )
        if self.check_if_best(score_summary):
            self.best_prediction_data = self.last_prediction_data.copy()
        self.log_monitor(score_summary)
        self.optuna_log_and_prune(score_summary)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        train_loss = mean(self.training_step_losses)
        self.training_step_losses.clear()
        self.wandb_log({"loss": train_loss}, "summary/train_")
        self.wandb_push_table()

    def on_fit_end(self):
        super().on_fit_end()

        self.wandb_add_preds()
        self.wandb_push_table(force=True)
        self.wandb_log_ckpt_files()

    def on_test_start(self) -> None:
        super().on_test_start()

        self.log_configuration()

    def on_test_end(self) -> None:
        super().on_test_end()

        test_loss = mean(self.test_step_losses)
        self.test_step_losses.clear()
        test_score = self.metric.compute()
        test_score = self.metric.prepare_for_log(test_score)

        self.wandb_log({"loss": test_loss} | dict(test_score), "summary/test_")
        self.best_prediction_data = self.last_prediction_data.copy()
        self.wandb_add_preds()
        self.wandb_push_table(force=True)

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.train_datasets)

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.val_datasets)

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.test_datasets)

    def init(
        self,
        resume: bool = False,
        force_clear_dir: bool = False,
        git_hash: str | None = None,
    ) -> bool:
        self.resume = resume
        if self.log_path == "":
            pass
        elif resume:
            ok = os.path.exists(self.log_path)
            if not ok:
                print("No data from previous learning")
                return False
        else:
            ok = check_rmtree(self.log_path, force_clear_dir)
            if not ok:
                print("Fit canceled")
                return False
            check_mkdir(FILENAMES["log_folder"])
            check_mkdir(self.log_path)

        self.wandb_init()

        if git_hash is None:
            git_hash = get_short_git_hash()
        print("-" * 30)
        print("Git hash:", git_hash)
        print("Command:", " ".join(sys.argv))

        self.init_ok = True
        return True

    def wandb_init(self):
        if not self.use_wandb:
            return

        assert wandb.run is not None, "Wandb run is not initialized"

        wandb_config = self.config.get("wandb")
        assert wandb_config is not None

        datasets: list[tuple[str, list[DatasetClass]]] = [
            ("train_dataset", self.train_datasets),
            ("val_dataset", self.val_datasets),
            ("test_dataset", self.test_datasets),
        ]
        for use_as, dataset_list in datasets:
            if dataset_list is None:
                continue
            if len(dataset_list) == 1:
                wandb.run.use_artifact(
                    f"{dataset_list[0].dataset_name}:latest",
                    type="dataset",
                    use_as=use_as,
                )
                continue
            for i, ds in enumerate(dataset_list):
                wandb.run.use_artifact(
                    f"{ds.dataset_name}:latest",
                    type="dataset",
                    use_as=f"{use_as}_{i}",
                )

        if wandb_config.get("watch_model"):
            wandb.watch(self, log_freq=1)

        wandb.define_metric("epoch")
        wandb.define_metric("summary/*", step_metric="epoch")

    def check_and_clean_config(self, ref_type: Type[ConfigUnion]):
        config = self.config.copy()
        keys_to_remove = []
        for key in config.keys():
            if key not in ref_type.__annotations__.keys():
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del config[key]
        for key in ref_type.__required_keys__:
            if key not in config.keys():
                raise ValueError(f"Missing {key} in config")
        self.config = config

    def make_dataset(
        self,
        mode: DatasetModes,
        datasets: list[tuple[Type[DatasetClass], DatasetKwargs]],
    ) -> list[DatasetClass]:
        return [
            dataset_class(
                mode,
                self.config["data"]["num_classes"],
                self.config["data"]["resize_to"],
                **dataset_kwargs,
            )
            for dataset_class, dataset_kwargs in datasets
        ]

    def prepare_datasets(self):
        start_time = time.perf_counter()
        for ds in self.train_datasets:
            ds.fill_cached_items_data()
        inter_time = time.perf_counter()
        self.print(f"train datasets prep done in {(inter_time - start_time):.2f} s")
        for ds in self.val_datasets:
            ds.fill_cached_items_data()
        end_time = time.perf_counter()
        self.print(f"val datasets prep done in {(end_time - inter_time):.2f} s")

    def get_dataset_item(
        self, type: Literal["TR", "VL", "TS"], dataset_name: str, index: int
    ) -> BaseDataTuple | None:
        if type == "TR":
            datasets = self.train_datasets
        elif type == "VL":
            datasets = self.val_datasets
        elif type == "TS":
            datasets = self.test_datasets
        else:
            raise ValueError(f"Invalid dataset type: {type}")
        for ds in datasets:
            if ds.dataset_name == dataset_name:
                return ds.get_data(index)
        return None

    def cast_example_input_array(self):
        if isinstance(self.example_input_array, tuple):
            self.example_input_array = tuple(
                inp.to(self.device) if isinstance(inp, Tensor) else inp
                for inp in self.example_input_array
            )
        elif isinstance(self.example_input_array, Tensor):
            self.example_input_array = self.example_input_array.to(self.device)

    def get_optimizer_list(self) -> list[LightningOptimizer]:
        opt = self.optimizers()
        if isinstance(opt, list):
            return opt
        return [opt]

    def get_scheduler_list(self) -> list[Scheduler]:
        sched = self.lr_schedulers()
        if sched is None:
            return []
        if isinstance(sched, list):
            return sched
        return [sched]

    def update_progress_bar_fields(self, task: ProgressBarTaskType, **kwargs):
        if isinstance(self.trainer.progress_bar_callback, CustomRichProgressBar):
            self.trainer.progress_bar_callback.update_fields(task, **kwargs)

    @contextmanager
    def profile(self, name: str) -> Generator:
        if self.config["learn"].get("profiler") is None:
            yield
            return
        if not hasattr(self.trainer, "profiler"):
            yield
            return
        profiler = self.trainer.profiler  # type: ignore
        if not isinstance(profiler, Profiler):
            yield
            return
        action_name = f"[Learner]{self.__class__.__name__}.{name}"
        with profiler.profile(action_name) as p:
            yield p

    def check_if_best(self, value: float) -> bool:
        monitor_mode = self.config["callbacks"].get("monitor_mode", "min")
        return (
            (self.best_monitor_value is None)
            or (monitor_mode == "min" and value < self.best_monitor_value)
            or (monitor_mode == "max" and value > self.best_monitor_value)
        )

    def get_configuration(self) -> dict:
        def serialize_datasets(
            datasets: list[tuple[Type[DatasetClass], DatasetKwargs]],
        ):
            return [
                {"class": get_name_from_class(cls), "kwargs": kwargs}
                for cls, kwargs in datasets
            ]

        optimizer_classes, scheduler_classes = get_optimizer_and_scheduler_names(
            self.configure_optimizers()
        )

        configuration = {
            "config": self.config,
            "optimizer_classes": optimizer_classes,
            "scheduler_classes": scheduler_classes,
            "loss_class": get_name_from_instance(self.loss),
            "loss_kwargs": self.loss_kwargs,
            "metric_class": get_name_from_instance(self.metric),
            "metric_data": self.metric_kwargs,
            "datasets": serialize_datasets(self.dataset_list),
        }
        if self.val_dataset_list is not None:
            configuration["val_datasets"] = serialize_datasets(self.val_dataset_list)
        if self.test_dataset_list is not None:
            configuration["test_datasets"] = serialize_datasets(self.test_dataset_list)

        return configuration

    def log_configuration(self):
        if self.optuna_trial:
            if self.use_wandb and wandb.run:
                study_id = self.optuna_trial.study.study_name.split(" ")[-1]
                wandb.run.use_artifact(
                    f"{prepare_study_ref_artifact_name(study_id)}:latest",
                    type="study-reference",
                )
        if not self.config["log"].get("configuration") or self.configuration_logged:
            return
        dummy = self.config["learn"].get("dummy") is True

        configuration = self.get_configuration()
        filepath = os.path.join(self.log_path, FILENAMES["configuration"])
        artifact_name = prepare_artifact_name(
            self.config["learn"]["exp_name"], self.config["learn"]["run_name"], "conf"
        )
        if self.resume:
            old_configuration = load_json(filepath)
            assert isinstance(old_configuration, dict)

            diff_filepath = os.path.join(self.log_path, FILENAMES["configuration_diff"])
            if os.path.isfile(diff_filepath):
                prev_diff_list = load_json(diff_filepath)
                assert isinstance(prev_diff_list, list)
            else:
                prev_diff_list = []

            new_diff = {"timestamp": get_iso_timestamp_now()}
            new_diff.update(diff_dict(old_configuration, configuration))
            new_diff_list = prev_diff_list + [new_diff]

            dump_json(diff_filepath, new_diff_list)
            if self.use_wandb:
                wandb_delete_files(
                    artifact_name,
                    "configuration",
                    excluded_aliases=["base"],
                    dummy=dummy,
                )
                wandb_log_file(wandb.run, artifact_name, diff_filepath, "configuration")
        else:
            dump_json(filepath, configuration)
            if self.use_wandb:
                wandb_log_file(
                    wandb.run, artifact_name, filepath, "configuration", ["base"]
                )

        if dummy:
            dummy_file = open(os.path.join(self.log_path, FILENAMES["dummy_file"]), "w")
            dummy_file.close()

        self.configuration_logged = True

    def log_tensorboard_graph(self):
        if not self.config["log"].get("tensorboard_graph"):
            return
        exp_name = self.config["learn"]["exp_name"]
        run_name = self.config["learn"]["run_name"]
        tensorboard_writer = SummaryWriter(self.log_path)
        tensorboard_writer.add_graph(self, self.example_input_array)
        tensorboard_writer.close()
        tb_files = sorted(filter(lambda x: "tfevents" in x, os.listdir(self.log_path)))
        if self.use_wandb and self.config.get("wandb", {}).get("save_model"):
            wandb_log_file(
                wandb.run,
                prepare_artifact_name(exp_name, run_name, "tb"),
                os.path.join(self.log_path, tb_files[-1]),
                "tensorboard_graph",
            )

    def log_model_onnx(self):
        if not self.config["log"].get("model_onnx"):
            return
        onnx_path = os.path.join(self.log_path, FILENAMES["model_onnx"])
        self.to_onnx(onnx_path, export_params=False)
        if self.use_wandb and self.config.get("wandb", {}).get("save_model"):
            artifact = wandb_log_file(
                wandb.run, self.__class__.__name__, onnx_path, "model"
            )
            if artifact and wandb.run:
                wandb.run.use_artifact(artifact)

    def log_table(
        self,
        data: list[tuple[str, Primitives]],
        group: str,
    ):
        if not self.config["log"].get("table"):
            return

        self.wandb_add_table(data, group)

        csv_filename = os.path.join(self.log_path, f"{group}.csv")
        write_to_csv(csv_filename, data)

    def log_monitor(self, value: float):
        if self.check_if_best(value):
            self.best_monitor_value = value

        name = self.config["callbacks"].get("monitor")
        if name is not None:
            self.log(name, value, on_step=False, on_epoch=True, batch_size=1)

    def wandb_log(self, data: Mapping[str, Primitives], prefix: str = ""):
        if not self.use_wandb:
            return
        use_epoch = prefix.startswith("summary/")
        epoch_value = 0 if "test" in prefix else self.current_epoch
        wandb.log(
            {prefix + k: v for k, v in data.items()}
            | ({"epoch": epoch_value} if use_epoch else {})
        )

    def wandb_log_ckpt_files(self):
        if (
            not self.use_wandb
            or self.log_path == ""
            or not self.config.get("wandb", {}).get("save_model")
        ):
            return

        artifact_name = prepare_ckpt_artifact_name(
            self.config["learn"]["exp_name"], self.config["learn"]["run_name"]
        )
        if self.resume:
            wandb_delete_files(
                artifact_name,
                "checkpoint",
                dummy=self.config["learn"].get("dummy") is True,
            )

        for ckpt in sorted(os.listdir(self.log_path)):
            if not ckpt.endswith(".ckpt"):
                continue
            artifact_alias = prepare_ckpt_artifact_alias(ckpt)
            wandb_log_file(
                wandb.run,
                artifact_name,
                os.path.join(self.log_path, ckpt),
                "checkpoint",
                [artifact_alias],
            )

    def wandb_add_table(
        self,
        data: Sequence[tuple[str, Primitives | wandb.Image]],
        group: str,
    ):
        use_wandb_table = self.config.get("wandb", {}).get("push_table_freq")
        if not self.use_wandb or not use_wandb_table:
            return
        if group not in self.wandb_tables:
            self.wandb_tables[group] = wandb.Table(columns=[d[0] for d in data])
        self.wandb_tables[group].add_data(*[d[1] for d in data])

    def wandb_push_table(self, force: bool = False):
        push_table_freq = self.config.get("wandb", {}).get("push_table_freq")
        if push_table_freq and (self.current_epoch % push_table_freq == 0 or force):
            wandb.log({k: t for k, t in self.wandb_tables.items() if len(t.data) > 0})
            for group in self.wandb_tables:
                self.wandb_tables[group] = wandb.Table(
                    columns=self.wandb_tables[group].columns
                )

    def wandb_add_mask(
        self,
        type: Literal["TR", "VL", "TS"],
        pred: Tensor,
        index: int,
        dataset: str,
        aux_data: list[tuple[str, Primitives]],
    ):
        if not self.use_wandb:
            return

        dataset_item = self.get_dataset_item(type, dataset, index)
        if dataset_item is None:
            return
        img, msk, file = dataset_item

        if pred.is_floating_point():
            pred = pred.argmax(dim=0)
        scores = self.metric.prepare_for_log(
            self.metric.measure(
                pred, BaseDataset.prepare_mask_as_tensor(msk).to(self.device)
            )
        )

        if self.config.get("wandb", {}).get("save_mask_only"):
            img = np.ones_like(msk) * 255

        wandb_image = wandb.Image(
            img,
            masks={
                "truth": {
                    "mask_data": msk,
                    "class_labels": self.class_labels,
                },
                "pred": {
                    "mask_data": pred.cpu().numpy(),
                    "class_labels": self.class_labels,
                },
            },
        )

        self.wandb_add_table(
            [("image", wandb_image)]
            + scores
            + [("type", type), ("dataset", dataset), ("file", file)]
            + aux_data,
            f"preds/{type}",
        )

    def wandb_handle_preds(
        self,
        type: Literal["TR", "VL", "TS"],
        batch_idx: int,
        preds: Tensor,
        indices: int | list[int],
        datasets: str | list[str],
        **kwargs: Primitives | ListPrimitives,
    ):
        if not self.use_wandb:
            return

        match type:
            case "TR":
                indices_to_save = self.train_indices_to_save
            case "VL":
                indices_to_save = self.val_indices_to_save
            case "TS":
                indices_to_save = self.test_indices_to_save
        if indices_to_save is None:
            return

        if batch_idx == 0:
            self.last_prediction_data[type] = []
        for i in indices_to_save[batch_idx]:
            dataset = datasets[i] if isinstance(datasets, list) else datasets
            index = indices[i] if isinstance(indices, list) else indices
            aux_data: list[tuple[str, Primitives]] = []
            for k, v in kwargs.items():
                if isinstance(v, list):
                    aux_data.append((k, v[i]))
                else:
                    aux_data.append((k, v))
            self.last_prediction_data[type].append((preds[i], index, dataset, aux_data))

    def wandb_add_preds(self):
        for type, pred_data in self.best_prediction_data.items():
            for pred, index, dataset, aux_data in pred_data:
                self.wandb_add_mask(
                    type,  # type: ignore
                    pred,
                    index,
                    dataset,
                    aux_data,
                )
        self.best_prediction_data = {}

    def optuna_log_and_prune(self, value: float):
        if self.optuna_trial is None:
            return
        self.optuna_trial.report(value, self.current_epoch)
        if self.optuna_trial.should_prune():
            self.optuna_pruned = True
            self.trainer.should_stop = True
