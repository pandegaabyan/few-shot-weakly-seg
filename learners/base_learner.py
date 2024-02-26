import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Type

import numpy as np
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import Unpack

import wandb
from config.config_type import ConfigUnion
from config.constants import FILENAMES, WANDB_SETTINGS
from data.typings import DatasetModes
from learners.losses import CustomLoss
from learners.metrics import CustomMetric
from learners.typings import (
    BaseLearnerKwargs,
    ConfigType,
    DatasetClass,
    DatasetKwargs,
)
from utils.logging import (
    check_mkdir,
    check_rmtree,
    dump_json,
    get_name_from_class,
    get_name_from_instance,
    get_short_git_hash,
    get_simple_stack_list,
    load_json,
    prepare_ckpt_path_for_artifact,
    write_to_csv,
)
from utils.utils import (
    diff_dict,
    get_iso_timestamp_now,
    mean,
    merge_dicts,
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

        self.loss = kwargs.get("loss") or CustomLoss()
        self.metric = kwargs.get("metric") or CustomMetric()
        self.resume = kwargs.get("resume", False)
        self.force_clear_dir = kwargs.get("force_clear_dir", False)
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

    def init(self) -> bool:
        if self.resume:
            ok = self.check_log_and_ckpt_dir()
            if not ok:
                print("No data from previous learning")
                return False
        else:
            ok = self.clear_log_and_ckpt_dir()
            if not ok:
                print("Fit canceled")
                return False
            self.create_log_and_ckpt_dir()

        if not self.config["learn"].get("ref_ckpt_path"):
            onnx_path = os.path.join(self.ckpt_path, FILENAMES["model_onnx"])
            self.to_onnx(onnx_path, export_params=False)

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

        dummy = self.config["learn"].get("dummy")
        parent_path = (
            WANDB_SETTINGS["entity"]
            + "/"
            + WANDB_SETTINGS["dummy_project" if dummy else "project"]
        )
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
                    f"{parent_path}/{ds.dataset_name}:latest",
                    type="dataset",
                    use_as=use_as,
                )
        ref_ckpt_path = self.config["learn"].get("ref_ckpt_path")
        if ref_ckpt_path:
            ckpt_name, ckpt_alias = prepare_ckpt_path_for_artifact(ref_ckpt_path)
            wandb.run.use_artifact(
                f"{parent_path}/{ckpt_name}:{ckpt_alias}", type="checkpoint"
            )

        if wandb_config["log_model"]:
            onnx_path = os.path.join(self.ckpt_path, FILENAMES["model_onnx"])
            model_artifact = wandb.Artifact(self.__class__.__name__, type="model")
            model_artifact.add_file(onnx_path)
            wandb.log_artifact(model_artifact)
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

    def check_log_and_ckpt_dir(self) -> bool:
        return os.path.exists(self.log_path) and os.path.exists(self.ckpt_path)

    def create_log_and_ckpt_dir(self):
        check_mkdir(FILENAMES["log_folder"])
        check_mkdir(self.log_path)
        check_mkdir(FILENAMES["checkpoint_folder"])
        check_mkdir(self.ckpt_path)

    def clear_log_and_ckpt_dir(self) -> bool:
        log_ok = check_rmtree(self.log_path, self.force_clear_dir)
        ckpt_ok = check_rmtree(self.ckpt_path, self.force_clear_dir)
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

    def log_configuration(self):
        def dictify_datasets(datasets: list[tuple[Type[DatasetClass], DatasetKwargs]]):
            if datasets is None:
                return []
            return [
                {"class": get_name_from_class(cls), "kwargs": kwargs}
                for cls, kwargs in datasets
            ]

        configuration = {
            "config": self.config,
            "datasets": dictify_datasets(self.dataset_list),
        }
        if self.val_dataset_list is not None:
            configuration["val_datasets"] = dictify_datasets(self.val_dataset_list)
        if self.test_dataset_list is not None:
            configuration["test_datasets"] = dictify_datasets(self.test_dataset_list)
        configuration.update(
            {
                "loss": get_name_from_instance(self.loss),
                "loss_data": self.loss.params(),
                "metric": get_name_from_instance(self.metric),
                "metric_data": self.metric.params(),
            }
        )

        filepath = os.path.join(self.log_path, FILENAMES["configuration"])
        if self.resume:
            diff_filepath = os.path.join(self.log_path, FILENAMES["configuration_diff"])
            old_configuration = load_json(filepath)
            assert isinstance(old_configuration, dict)
            if os.path.isfile(diff_filepath):
                prev_diff_list = load_json(diff_filepath)
                assert isinstance(prev_diff_list, list)
            else:
                prev_diff_list = []
            new_diff = {"timestamp": get_iso_timestamp_now()}
            new_diff.update(diff_dict(old_configuration, configuration))
            new_diff_list = prev_diff_list + [new_diff]
            dump_json(diff_filepath, new_diff_list)
        else:
            dump_json(filepath, configuration)

        if self.config["learn"].get("dummy") is True:
            dummy_file = open(os.path.join(self.log_path, FILENAMES["dummy_file"]), "w")
            dummy_file.close()

    def log_tensorboard_graph(self):
        if not self.tensorboard_graph:
            return
        tensorboard_dir = os.path.join(
            FILENAMES["tensorboard_folder"],
            self.config["learn"]["exp_name"],
            self.config["learn"]["run_name"],
        )
        check_rmtree(tensorboard_dir, True)
        tensorboard_writer = SummaryWriter(tensorboard_dir)
        tensorboard_writer.add_graph(self, self.example_input_array)
        tensorboard_writer.close()

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

    def wandb_log_ckpt_files(self):
        if (
            not self.use_wandb
            or self.current_epoch != self.config["learn"]["num_epochs"] - 1
        ):
            return

        for ckpt in os.listdir(self.ckpt_path):
            if not ckpt.endswith(".ckpt"):
                continue
            ckpt_path = f"{self.config['learn']['exp_name']}/{self.config['learn']['run_name']}/{ckpt}"
            name, alias = prepare_ckpt_path_for_artifact(ckpt_path)
            artifact = wandb.Artifact(name, type="checkpoint")
            artifact.add_reference(
                "file://"
                + os.path.join(
                    os.getcwd().replace("\\", "/"),
                    FILENAMES["checkpoint_folder"].removeprefix("./"),
                    ckpt_path,
                ),
                checksum=False,
            )
            wandb.log_artifact(artifact, aliases=[alias])

    def log_to_wandb_preds(
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
