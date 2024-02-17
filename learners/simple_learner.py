import time
from abc import ABC, abstractmethod
from typing import Any, Type

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from config.config_type import ConfigSimpleLearner
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.losses import CustomLoss
from learners.metrics import CustomMetric
from learners.typings import SimpleDataBatchTuple
from utils.logging import get_count_as_text


class SimpleLearner(
    BaseLearner[ConfigSimpleLearner, SimpleDataset, SimpleDatasetKwargs], ABC
):
    def __init__(
        self,
        config: ConfigSimpleLearner,
        dataset_list: list[tuple[Type[SimpleDataset], SimpleDatasetKwargs]],
        val_dataset_list: list[tuple[Type[SimpleDataset], SimpleDatasetKwargs]]
        | None = None,
        test_dataset_list: list[tuple[Type[SimpleDataset], SimpleDatasetKwargs]]
        | None = None,
        loss: CustomLoss | None = None,
        metric: CustomMetric | None = None,
        resume: bool = False,
        force_clear_dir: bool = False,
    ):
        super().__init__(
            config,
            dataset_list,
            val_dataset_list,
            test_dataset_list,
            loss,
            metric,
            resume,
            force_clear_dir,
        )

        self.check_and_clean_config(ConfigSimpleLearner)

        self.net = self.make_net()
        self.training_step_losses = []
        self.validation_step_losses = []
        self.test_step_losses = []

    @abstractmethod
    def make_net(self) -> nn.Module:
        pass

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        start_time = time.perf_counter()
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)
        self.training_step_losses.append(loss.item())
        end_time = time.perf_counter()

        self.log_table(
            [
                ("batch", batch_idx),
                ("loss", loss.item()),
                ("duration", end_time - start_time),
                ("dataset", get_count_as_text(dataset_names)),
            ],
            "train",
        )
        self.wandb_log({"loss": loss.item()}, "train/")

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            for i in self.train_indices_to_save[batch_idx]:
                self.wandb_log_image(
                    mask[i],
                    pred[i],
                    [("file_name", file_names[i]), ("dataset", dataset_names[i])],
                    "train_preds",
                )

        return loss

    def validation_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        val_freq = self.config["learn"].get("val_freq", 1)
        if val_freq != 0 and self.current_epoch % val_freq != 0:
            return

        start_time = time.perf_counter()
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)
        self.validation_step_losses.append(loss.item())
        score = self.metric(pred, mask)
        score = self.metric.prepare_for_log(score)
        end_time = time.perf_counter()

        if self.trainer.sanity_checking:
            return loss

        self.log_table(
            [
                ("batch", batch_idx),
                ("loss", loss.item()),
                ("duration", end_time - start_time),
                ("dataset", get_count_as_text(dataset_names)),
            ]
            + score,
            "val",
        )
        self.wandb_log({"loss": loss.item()} | dict(score), "val/")

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            for i in self.val_indices_to_save[batch_idx]:
                self.wandb_log_image(
                    mask[i],
                    pred[i],
                    [("file_name", file_names[i]), ("dataset", dataset_names[i])],
                    "val_preds",
                )

        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        train_loss = sum(self.training_step_losses) / len(self.training_step_losses)
        self.log_table(
            [("loss", train_loss)],
            "train_summary",
            index="epoch",
        )
        self.wandb_log({"loss": train_loss}, "summary/train_", True)
        self.training_step_losses.clear()

        total_epoch = self.config["learn"]["num_epochs"] - 1
        message = (
            f"Epoch {self.current_epoch}/{total_epoch}  train_loss: {train_loss:.4f}"
        )

        if len(self.validation_step_losses) != 0:
            val_loss = sum(self.validation_step_losses) / len(
                self.validation_step_losses
            )
            val_score = self.metric.compute()
            val_score = self.metric.prepare_for_log(val_score)
            score_summary = self.metric.score_summary()
            self.log_table(
                [("loss", val_loss)] + val_score,
                "val_summary",
                index="epoch",
            )
            self.wandb_log(
                {"loss": val_loss} | dict(val_score) | {"score": score_summary},
                "summary/val_",
                True,
            )
            self.log_checkpoint_ref(score_summary)
            self.validation_step_losses.clear()
            message += f"  val_loss: {val_loss:.4f}  val_score: {dict(val_score)}"

        self.print(message)
        self.wandb_push_table()
        self.wandb_log_ckpt_ref()

    def test_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        start_time = time.perf_counter()
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)
        self.test_step_losses.append(loss.item())
        score = self.metric(pred, mask)
        score = self.metric.prepare_for_log(score)
        end_time = time.perf_counter()

        self.log_table(
            [
                ("batch", batch_idx),
                ("loss", loss.item()),
                ("duration", end_time - start_time),
                ("dataset", get_count_as_text(dataset_names)),
            ]
            + score,
            "test",
            index="step",
        )
        self.wandb_log({"loss": loss.item()} | dict(score), "test/")

        for i in self.test_indices_to_save[batch_idx]:
            self.wandb_log_image(
                mask[i],
                pred[i],
                [("file_name", file_names[i]), ("dataset", dataset_names[i])],
                "test_preds",
            )

        return loss

    def on_test_end(self) -> None:
        super().on_test_end()
        test_loss = sum(self.test_step_losses) / len(self.test_step_losses)
        test_score = self.metric.compute()
        test_score = self.metric.prepare_for_log(test_score)
        self.log_table(
            [("loss", test_loss)] + test_score,
            "test_summary",
            index="epoch",
        )
        self.wandb_log({"loss": test_loss} | dict(test_score), "summary/test_", True)
        self.test_step_losses.clear()
        self.print(f"Test  loss: {test_loss}  score: {dict(test_score)}")
        self.wandb_push_table()

    def make_dataloader(self, datasets: list[SimpleDataset]):
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=self.config["data"]["batch_size"],
            shuffle=datasets[0].mode == "train",
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.device.type != "cpu",
        )

    def make_input_example(self) -> tuple[Any, ...]:
        input_example = torch.rand(
            self.config["data"]["batch_size"],
            self.config["data"]["num_channels"],
            *self.config["data"]["resize_to"],
        )
        return (input_example,)
