from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader

from config.config_type import ConfigSimpleLearner
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.typings import SimpleDataBatchTuple
from utils.utils import make_batch_sample_indices


class SimpleLearner(
    BaseLearner[ConfigSimpleLearner, SimpleDataset, SimpleDatasetKwargs], ABC
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigSimpleLearner)

        self.net = self.make_net()

    @abstractmethod
    def make_net(self) -> nn.Module:
        pass

    def make_dataloader(self, datasets: list[SimpleDataset]):
        mode = datasets[0].mode
        num_workers = self.config["data"]["num_workers"]
        batch_size = 1 if mode == "test" else self.config["data"]["batch_size"]
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.device.type != "cpu",
        )

    def make_indices_to_save(
        self, datasets: list[SimpleDataset], sample_size: int
    ) -> list[list[int]] | None:
        mode = datasets[0].mode
        batch_size = 1 if mode == "test" else self.config["data"]["batch_size"]
        return make_batch_sample_indices(
            sum(len(ds) for ds in datasets),
            sample_size,
            batch_size,
        )

    def make_input_example(self) -> tuple[Any, ...]:
        image_example = torch.rand(
            self.config["data"]["batch_size"],
            self.config["data"]["num_channels"],
            *self.config["data"]["resize_to"],
        )
        return (image_example,)

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)

    def training_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        with self.profile("forward:training"):
            pred = self.forward(image)
        loss = self.loss(pred, mask)

        self.training_step_losses.append(loss.item())

        self.log_to_table_metrics("TR", batch_idx, loss)

        self.wandb_handle_preds("TR", batch_idx, mask, pred, file_names, dataset_names)

        return loss

    def validation_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        with self.profile("forward:validation"):
            pred = self.forward(image)
        loss = self.loss(pred, mask)

        self.validation_step_losses.append(loss.item())
        score = self.metric(pred, mask)

        if self.trainer.sanity_checking:
            return loss

        self.log_to_table_metrics("VL", batch_idx, loss, score=score)

        self.wandb_handle_preds("VL", batch_idx, mask, pred, file_names, dataset_names)

        return loss

    def test_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, file_names, dataset_names = batch
        with self.profile("forward:test"):
            pred = self.forward(image)
        loss = self.loss(pred, mask)

        self.test_step_losses.append(loss.item())
        score = self.metric(pred, mask)

        self.log_to_table_metrics("TS", batch_idx, loss, score=score, epoch=0)

        self.wandb_handle_preds("TS", batch_idx, mask, pred, file_names, dataset_names)

        return loss

    def log_to_table_metrics(
        self,
        type: Literal["TR", "VL", "TS"],
        batch_idx: int,
        loss: Tensor,
        score: dict[str, Tensor] | None = None,
        epoch: int | None = None,
    ):
        if not self.config["log"].get("table"):
            return
        if score is not None:
            score_tup = self.metric.prepare_for_log(score)
        else:
            score_tup = [(key, None) for key in sorted(self.metric.metrics)]
        self.log_table(
            [
                ("type", type),
                ("epoch", epoch if epoch is not None else self.current_epoch),
                ("batch", batch_idx),
                ("loss", loss.item()),
            ]
            + score_tup,
            "metrics",
        )
