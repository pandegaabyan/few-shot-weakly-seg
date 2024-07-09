import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Type

import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Unpack

import wandb
from config.config_type import ConfigUnion
from config.constants import FILENAMES
from data.typings import DatasetModes
from learners.losses import CustomLoss
from learners.metrics import MultiIoUMetric
from learners.typings import (
    BaseLearnerKwargs,
    ConfigType,
    DatasetClass,
    DatasetKwargs,
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
    get_simple_stack_list,
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
    prepare_ckpt_artifact_name,
    wandb_download_file,
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

        self.train_datasets = self.make_dataset("train", self.dataset_list)
        self.val_datasets = self.make_dataset(
            "val", self.val_dataset_list or self.dataset_list
        )
        self.test_datasets = self.make_dataset(
            "test", self.test_dataset_list or self.dataset_list
        )

        (loss_class, loss_kwargs) = kwargs.get("loss") or (CustomLoss, {})
        self.loss = loss_class(**loss_kwargs)
        self.loss_kwargs = loss_kwargs
        (metric_class, metric_kwargs) = kwargs.get("metric") or (MultiIoUMetric, {})
        self.metric = metric_class(**metric_kwargs)
        self.metric_kwargs = metric_kwargs

        self.use_wandb = self.config.get("wandb") is not None
        self.tensorboard_graph = self.config["learn"].get("tensorboard_graph", True)

        self.initial_messages: list[str] = []
        self.log_path = os.path.join(
            FILENAMES["log_folder"],
            self.config["learn"]["exp_name"],
            self.config["learn"]["run_name"],
        )
        self.ckpt_path = os.path.join(
            FILENAMES["checkpoint_folder"],
            self.config["learn"]["exp_name"],
            self.config["learn"]["run_name"],
        )

        wandb_config = self.config.get("wandb", {})
        self.train_indices_to_save = self.make_indices_to_save(
            self.train_datasets, wandb_config.get("save_train_preds")
        )
        self.val_indices_to_save = self.make_indices_to_save(
            self.val_datasets, wandb_config.get("save_val_preds")
        )
        self.test_indices_to_save = self.make_indices_to_save(
            self.test_datasets, wandb_config.get("save_test_preds")
        )
        self.class_labels = merge_dicts(
            [
                ds.class_labels
                for ds in (self.train_datasets + self.val_datasets + self.test_datasets)
            ]
        )

        self.init_ok = False
        self.resume = False
        self.example_input_array = self.make_input_example()

        self.wandb_tables: dict[str, wandb.Table] = {}
        self.training_step_losses: list[float] = []
        self.validation_step_losses: list[float] = []
        self.test_step_losses: list[float] = []

    @abstractmethod
    def make_dataloader(self, datasets: list[DatasetClass]) -> DataLoader:
        pass

    @abstractmethod
    def make_indices_to_save(
        self, datasets: list[DatasetClass], sample_size: int | None
    ) -> list[list[int]]:
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
        self.log_monitor(score_summary)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        train_loss = mean(self.training_step_losses)
        self.training_step_losses.clear()
        self.wandb_log({"loss": train_loss}, "summary/train_")

        self.wandb_push_table()
        self.wandb_log_ckpt_files()

    def on_test_end(self) -> None:
        super().on_test_end()

        test_loss = mean(self.test_step_losses)
        self.test_step_losses.clear()
        test_score = self.metric.compute()
        test_score = self.metric.prepare_for_log(test_score)

        self.wandb_log({"loss": test_loss} | dict(test_score), "summary/test_")

        self.wandb_push_table(force=True)

    def train_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.train_datasets)

    def val_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.val_datasets)

    def test_dataloader(self) -> DataLoader:
        return self.make_dataloader(self.test_datasets)

    def init(self, resume: bool = False, force_clear_dir: bool = False) -> bool:
        self.resume = resume
        if resume:
            ok = self.check_log_and_ckpt_dir()
            if not ok:
                print("No data from previous learning")
                return False
        else:
            ok = self.clear_log_and_ckpt_dir(force_clear_dir)
            if not ok:
                print("Fit canceled")
                return False
            self.create_log_and_ckpt_dir()

        self.wandb_init()

        self.print_initial_info()
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
            for ds in dataset_list:
                wandb.run.use_artifact(
                    f"{ds.dataset_name}:latest",
                    type="dataset",
                    use_as=use_as,
                )
        ref_ckpt_path = self.config["learn"].get("ref_ckpt_path")
        if ref_ckpt_path:
            exp_run_name, ckpt_alias = prepare_ckpt_artifact_name(
                *ref_ckpt_path.split("/")
            )
            wandb.run.use_artifact(f"{exp_run_name}:{ckpt_alias}", type="checkpoint")

        if wandb_config["watch_model"]:
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
        self.print("Preparing train datasets ... ")
        start_time = time.perf_counter()
        for ds in self.train_datasets:
            ds.fill_cached_items_data()
        inter_time = time.perf_counter()
        self.print(f"train preparation done in {(inter_time - start_time):.2f} s")
        self.print("Preparing val datasets ... ")
        for ds in self.val_datasets:
            ds.fill_cached_items_data()
        end_time = time.perf_counter()
        self.print(f"val preparation done in {(end_time - inter_time):.2f} s")

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

    def check_log_and_ckpt_dir(self) -> bool:
        return os.path.exists(self.log_path) and os.path.exists(self.ckpt_path)

    def create_log_and_ckpt_dir(self):
        check_mkdir(FILENAMES["log_folder"])
        check_mkdir(self.log_path)
        check_mkdir(FILENAMES["checkpoint_folder"])
        check_mkdir(self.ckpt_path)

    def clear_log_and_ckpt_dir(self, force: bool = False) -> bool:
        log_ok = check_rmtree(self.log_path, force)
        ckpt_ok = check_rmtree(self.ckpt_path, force)
        return log_ok and ckpt_ok

    def print_initial_info(self):
        print("-" * 30)
        print("Git hash: " + get_short_git_hash())
        stack_list = get_simple_stack_list(end=-3)
        if any("ipykernel" in sl for sl in stack_list):
            print("Call stack: (called in jupyter notebook)")
        else:
            for i, stack in enumerate(stack_list):
                print(f"Call stack ({i}): {stack}")
        for msg in self.initial_messages:
            print("Note: " + msg)
        print("")

    def set_initial_messages(self, messages: list[str]):
        self.initial_messages = messages

    def update_progress_bar_fields(self, task: ProgressBarTaskType, **kwargs):
        if isinstance(self.trainer.progress_bar_callback, CustomRichProgressBar):
            self.trainer.progress_bar_callback.update_fields(task, **kwargs)

    def log_configuration(self):
        def serialize_datasets(
            datasets: list[tuple[Type[DatasetClass], DatasetKwargs]],
        ):
            return [
                {"class": get_name_from_class(cls), "kwargs": kwargs}
                for cls, kwargs in datasets
            ]

        optimizer_classes = [
            get_name_from_instance(opt) for opt in self.get_optimizer_list()
        ]
        scheduler_classes = [
            get_name_from_instance(sched) for sched in self.get_scheduler_list()
        ]

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

        filepath = os.path.join(self.log_path, FILENAMES["configuration"])
        artifact_name = prepare_artifact_name(
            self.config["learn"]["exp_name"], self.config["learn"]["run_name"], "conf"
        )
        if self.resume:
            if os.path.isfile(filepath) or (
                wandb_download_file(
                    wandb.run,
                    artifact_name + ":base",
                    self.log_path,
                    "configuration",
                    filepath,
                )
            ):
                old_configuration = load_json(filepath)
                assert isinstance(old_configuration, dict)
            else:
                raise FileNotFoundError(f"Configuration file not found: {filepath}")

            diff_filepath = os.path.join(self.log_path, FILENAMES["configuration_diff"])
            if os.path.isfile(diff_filepath) or (
                wandb_download_file(
                    wandb.run,
                    artifact_name + ":latest",
                    self.log_path,
                    "configuration",
                    diff_filepath,
                )
            ):
                prev_diff_list = load_json(diff_filepath)
                assert isinstance(prev_diff_list, list)
            else:
                prev_diff_list = []

            new_diff = {"timestamp": get_iso_timestamp_now()}
            new_diff.update(diff_dict(old_configuration, configuration))
            new_diff_list = prev_diff_list + [new_diff]

            dump_json(diff_filepath, new_diff_list)
            wandb_log_file(wandb.run, artifact_name, diff_filepath, "configuration")
        else:
            dump_json(filepath, configuration)
            wandb_log_file(
                wandb.run, artifact_name, filepath, "configuration", ["base"]
            )

        if self.config["learn"].get("dummy") is True:
            dummy_file = open(os.path.join(self.log_path, FILENAMES["dummy_file"]), "w")
            dummy_file.close()

    def log_tensorboard_graph(self):
        if not self.tensorboard_graph:
            return
        exp_name = self.config["learn"]["exp_name"]
        run_name = self.config["learn"]["run_name"]
        tensorboard_dir = os.path.join(
            FILENAMES["tensorboard_folder"], exp_name, run_name
        )
        check_rmtree(tensorboard_dir, True)
        tensorboard_writer = SummaryWriter(tensorboard_dir)
        tensorboard_writer.add_graph(self, self.example_input_array)
        tensorboard_writer.close()
        for file in os.listdir(tensorboard_dir):
            wandb_log_file(
                wandb.run,
                prepare_artifact_name(exp_name, run_name, "tb"),
                os.path.join(tensorboard_dir, file),
                "tensorboard_graph",
            )

    def log_model_onnx(self):
        if not self.config["learn"].get("model_onnx"):
            return
        if self.config["learn"].get("ref_ckpt_path"):
            return
        onnx_path = os.path.join(self.ckpt_path, FILENAMES["model_onnx"])
        self.to_onnx(onnx_path, export_params=False)
        artifact = wandb_log_file(
            wandb.run, self.__class__.__name__, onnx_path, "model"
        )
        if artifact and wandb.run:
            wandb.run.use_artifact(artifact)

    def log_table(
        self,
        data: list[tuple[str, Any]],
        group: str,
    ):
        self.wandb_log_table(data, group)

        csv_filename = os.path.join(self.log_path, f"{group}.csv")
        write_to_csv(csv_filename, data)

    def log_monitor(self, value: float):
        name = self.config["callbacks"].get("monitor")
        if name is not None:
            self.log(name, value, on_step=False, on_epoch=True, batch_size=1)

    def wandb_log(self, data: dict[str, Any], prefix: str = ""):
        if not self.use_wandb:
            return
        use_epoch = prefix.startswith("summary/")
        epoch_value = 0 if "test" in prefix else self.current_epoch
        wandb.log(
            {prefix + k: v for k, v in data.items()}
            | ({"epoch": epoch_value} if use_epoch else {})
        )

    def wandb_log_ckpt_files(self):
        if self.current_epoch != self.config["learn"]["num_epochs"] - 1:
            return

        for ckpt in os.listdir(self.ckpt_path):
            if not ckpt.endswith(".ckpt"):
                continue
            name, alias = prepare_ckpt_artifact_name(
                self.config["learn"]["exp_name"], self.config["learn"]["run_name"], ckpt
            )
            wandb_log_file(
                wandb.run,
                name,
                os.path.join(self.ckpt_path, ckpt),
                "checkpoint",
                [alias],
            )

    def wandb_log_table(
        self,
        data: list[tuple[str, Any]],
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
        last_epoch = self.current_epoch == self.config["learn"]["num_epochs"] - 1
        if push_table_freq and (
            self.current_epoch % push_table_freq == 0 or last_epoch or force
        ):
            wandb.log(self.wandb_tables)
            for group in self.wandb_tables:
                self.wandb_tables[group] = wandb.Table(
                    columns=self.wandb_tables[group].columns
                )

    def wandb_log_mask(
        self,
        gt: Tensor,
        pred: Tensor,
        data: list[tuple[str, Any]],
        group: str,
        image: Tensor | None = None,
    ):
        gt_arr = gt.cpu().numpy()
        pred_arr = pred.argmax(0).cpu().numpy()
        if image is not None:
            image_arr = np.moveaxis(image.cpu().numpy(), 0, -1).astype(np.uint8)
        else:
            image_arr = np.zeros(gt_arr.shape + (1,), dtype=np.uint8)

        gt_img = wandb.Image(
            image_arr,
            masks={
                "ground_truth": {
                    "mask_data": gt_arr,
                    "class_labels": self.class_labels,
                },
            },
        )
        pred_img = wandb.Image(
            image_arr,
            masks={
                "prediction": {
                    "mask_data": pred_arr,
                    "class_labels": self.class_labels,
                },
            },
        )
        img_data = [("ground_truth", gt_img), ("prediction", pred_img)]

        self.wandb_log_table(img_data + data, group)

    def wandb_log_preds(
        self,
        type: Literal["TR", "VL", "TS"],
        batch_idx: int,
        gt: Tensor,
        pred: Tensor,
        file_name: str | list[str],
        dataset: str | list[str],
    ):
        match type:
            case "TR":
                indices_to_save = self.train_indices_to_save
            case "VL":
                indices_to_save = self.val_indices_to_save
            case "TS":
                indices_to_save = self.test_indices_to_save
        for i in indices_to_save[batch_idx]:
            if isinstance(file_name, list):
                file_name = file_name[i]
            if isinstance(dataset, list):
                dataset = dataset[i]
            self.wandb_log_mask(
                gt[i],
                pred[i],
                [
                    ("type", "TR"),
                    ("file_name", file_name[i]),
                    ("dataset", dataset[i]),
                ],
                "preds",
            )
