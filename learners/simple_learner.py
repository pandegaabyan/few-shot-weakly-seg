from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader
from typing_extensions import Unpack

from config.config_type import ConfigSimpleLearner
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.typings import SimpleDataBatchTuple, SimpleLearnerKwargs
from utils.utils import make_batch_sample_indices


class SimpleLearner(
    BaseLearner[ConfigSimpleLearner, SimpleDataset, SimpleDatasetKwargs], ABC
):
    def __init__(self, **kwargs: Unpack[SimpleLearnerKwargs]):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigSimpleLearner)

        self.net = self.make_net()

    @abstractmethod
    def make_net(self) -> nn.Module:
        pass

    def make_dataloader(self, datasets: list[SimpleDataset]):
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=self.config["data"]["batch_size"],
            shuffle=datasets[0].mode == "train",
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.device.type != "cpu",
        )

    def make_indices_to_save(
        self, datasets: list[SimpleDataset], sample_size: int | None
    ) -> list[list[int]]:
        batch_size = self.config["data"]["batch_size"]
        return make_batch_sample_indices(
            sum(len(ds) for ds in datasets),
            sample_size or 0,
            batch_size,
        )

    def make_input_example(self) -> tuple[Any, ...]:
        img_example = torch.rand(
            self.config["data"]["batch_size"],
            self.config["data"]["num_channels"],
            *self.config["data"]["resize_to"],
        )
        return (img_example,)

    def forward(self, img: Tensor) -> Tensor:
        return self.net(img)

    def training_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)

        self.training_step_losses.append(loss.item())

        self.log_to_table_metrics("TR", batch_idx, loss)

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            self.log_to_wandb_preds(
                "TR", batch_idx, mask, pred, file_names, dataset_names
            )

        return loss

    def validation_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)

        self.validation_step_losses.append(loss.item())
        score = self.metric(pred, mask)
        score = self.metric.prepare_for_log(score)

        if self.trainer.sanity_checking:
            return loss

        self.log_to_table_metrics("VL", batch_idx, loss, score=score)

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            self.log_to_wandb_preds(
                "VL", batch_idx, mask, pred, file_names, dataset_names
            )

        return loss

    def test_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        pred = self(image)
        loss = self.loss(pred, mask)

        self.test_step_losses.append(loss.item())
        score = self.metric(pred, mask)
        score = self.metric.prepare_for_log(score)

        self.log_to_table_metrics("TS", batch_idx, loss, score=score, epoch=0)

        self.log_to_wandb_preds("TS", batch_idx, mask, pred, file_names, dataset_names)

        return loss

    def log_to_table_metrics(
        self,
        type: Literal["TR", "VL", "TS"],
        batch_idx: int,
        loss: Tensor,
        score: list[tuple[str, float]] | None = None,
        epoch: int | None = None,
    ):
        dummy_score = [(key, None) for key in sorted(self.metric.additional_params())]
        self.log_table(
            [
                ("type", type),
                ("epoch", epoch if epoch is not None else self.current_epoch),
                ("batch", batch_idx),
                ("loss", loss.item()),
            ]
            + (score if score is not None else dummy_score),
            "metrics",
        )
