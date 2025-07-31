from typing import Any, Literal

import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader

from config.config_type import ConfigSimpleLearner
from data.simple_dataset import SimpleDataset
from data.typings import SimpleDatasetKwargs
from learners.base_learner import BaseLearner
from learners.models import make_segmentation_model
from learners.optimizers import make_optimizer_adam, make_scheduler_step
from learners.typings import SimpleDataBatchTuple
from utils.utils import make_batch_sample_indices


class SimpleLearner(
    BaseLearner[ConfigSimpleLearner, SimpleDataset, SimpleDatasetKwargs]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.check_and_clean_config(ConfigSimpleLearner)

        self.net = self.make_net()

    def make_net(self) -> nn.Module:
        num_classes = self.config["data"]["num_classes"]
        output_channels = num_classes if num_classes != 2 else 1
        return make_segmentation_model(
            self.config["model"],
            self.config["data"]["num_channels"],
            output_channels,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        adam_optimizer = make_optimizer_adam(self.config["optimizer"], self.net)
        step_scheduler = make_scheduler_step(adam_optimizer, self.config["scheduler"])
        return [adam_optimizer], [step_scheduler]

    def make_dataloader(self, datasets: list[SimpleDataset]):
        mode = datasets[0].mode
        num_workers = self.config["data"]["num_workers"]
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=self.config["data"]["batch_size"],
            shuffle=mode == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.device.type != "cpu",
        )

    def make_indices_to_save(
        self, datasets: list[SimpleDataset], sample_size: int
    ) -> list[list[int]] | None:
        return make_batch_sample_indices(
            sum(len(ds) for ds in datasets),
            sample_size,
            self.config["data"]["batch_size"],
            seed=self.config["learn"].get("seed"),
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
        image, mask, _, _ = batch
        with self.profile("forward"):
            pred = self.forward(image)
        loss = self.loss(pred, mask)

        self.training_step_losses.append(loss.item())

        self.handle_metrics("TR", batch_idx, loss)

        self.handle_preds("TR", batch, batch_idx, pred)

        return loss

    def validation_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, _, _ = batch
        with self.profile("forward"):
            pred = self.forward(image)
        loss = self.loss(pred, mask)

        self.validation_step_losses.append(loss.item())
        score = self.metric(pred, mask)

        if self.trainer.sanity_checking:
            return loss

        self.handle_metrics("VL", batch_idx, loss, score=score)

        self.handle_preds("VL", batch, batch_idx, pred)

        return loss

    def test_step(self, batch: SimpleDataBatchTuple, batch_idx: int):
        image, mask, _, _ = batch
        with self.profile("forward"):
            pred = self.forward(image)
        loss = self.loss(pred, mask)

        self.test_step_losses.append(loss.item())
        score = self.metric(pred, mask)

        self.handle_metrics("TS", batch_idx, loss, score=score, epoch=0)

        self.handle_preds("TS", batch, batch_idx, pred)

        return loss

    def handle_metrics(
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

    def handle_preds(
        self,
        type: Literal["TR", "VL", "TS"],
        batch: SimpleDataBatchTuple,
        batch_idx: int,
        preds: Tensor,
    ):
        _, _, indices, datasets = batch
        self.wandb_handle_preds(type, batch_idx, preds, indices, datasets)
