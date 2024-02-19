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
from utils.utils import mean


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
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)

        self.training_step_losses.append(loss.item())
        dummy_score = [(key, None) for key in sorted(self.metric.additional_params())]

        self.log_table(
            [
                ("type", "TR"),
                ("epoch", self.current_epoch),
                ("batch", batch_idx),
                ("loss", loss.item()),
            ]
            + dummy_score,
            "metrics",
        )

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            for i in self.train_indices_to_save[batch_idx]:
                self.wandb_log_image(
                    mask[i],
                    pred[i],
                    [
                        ("type", "TR"),
                        ("file_name", file_names[i]),
                        ("dataset", dataset_names[i]),
                    ],
                    "preds",
                )

        return loss

    def validation_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        val_freq = self.config["learn"].get("val_freq", 1)
        if val_freq != 0 and self.current_epoch % val_freq != 0:
            return

        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)

        if self.trainer.sanity_checking:
            return loss

        self.validation_step_losses.append(loss.item())
        score = self.metric(pred, mask)
        score = self.metric.prepare_for_log(score)

        self.log_table(
            [
                ("type", "VL"),
                ("epoch", self.current_epoch),
                ("batch", batch_idx),
                ("loss", loss.item()),
            ]
            + score,
            "metrics",
        )

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            for i in self.val_indices_to_save[batch_idx]:
                self.wandb_log_image(
                    mask[i],
                    pred[i],
                    [
                        ("type", "VL"),
                        ("file_name", file_names[i]),
                        ("dataset", dataset_names[i]),
                    ],
                    "preds",
                )

        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        train_loss = mean(self.training_step_losses)
        self.training_step_losses.clear()
        self.wandb_log({"loss": train_loss}, "summary/train_")

        total_epoch = self.config["learn"]["num_epochs"] - 1
        message = (
            f"Epoch {self.current_epoch}/{total_epoch}  train_loss: {train_loss:.4f}"
        )

        if len(self.validation_step_losses) != 0:
            val_loss = mean(self.validation_step_losses)
            self.validation_step_losses.clear()
            val_score = self.metric.compute()
            val_score = self.metric.prepare_for_log(val_score)
            score_summary = self.metric.score_summary()

            self.wandb_log(
                {"loss": val_loss} | dict(val_score) | {"score": score_summary},
                "summary/val_",
            )
            self.log_checkpoint_ref(score_summary)
            message += f"  val_loss: {val_loss:.4f}  val_score: {dict(val_score)}"

        self.print(message)
        self.wandb_push_table()
        self.wandb_log_ckpt_ref()

    def test_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)

        self.test_step_losses.append(loss.item())
        score = self.metric(pred, mask)
        score = self.metric.prepare_for_log(score)

        self.log_table(
            [
                ("type", "TS"),
                ("epoch", 0),
                ("batch", batch_idx),
                ("loss", loss.item()),
            ]
            + score,
            "metrics",
        )

        for i in self.test_indices_to_save[batch_idx]:
            self.wandb_log_image(
                mask[i],
                pred[i],
                [
                    ("type", "TS"),
                    ("file_name", file_names[i]),
                    ("dataset", dataset_names[i]),
                ],
                "preds",
            )

        return loss

    def on_test_end(self) -> None:
        super().on_test_end()

        test_loss = mean(self.test_step_losses)
        self.test_step_losses.clear()
        test_score = self.metric.compute()
        test_score = self.metric.prepare_for_log(test_score)

        self.wandb_log({"loss": test_loss} | dict(test_score), "summary/test_")

        self.print(f"Test  loss: {test_loss}  score: {dict(test_score)}")
        self.wandb_push_table(force=True)

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
