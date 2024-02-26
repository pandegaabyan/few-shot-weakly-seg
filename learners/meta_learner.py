from abc import ABC, abstractmethod
from typing import Any, Generic, Literal

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from data.few_sparse_dataset import FewSparseDataset
from data.typings import FewSparseDatasetKwargs, FewSparseDataTuple, SupportDataTuple
from learners.base_learner import BaseLearner
from learners.typings import ConfigTypeMeta
from utils.utils import make_batch_sample_indices


class MetaLearner(
    Generic[ConfigTypeMeta],
    BaseLearner[ConfigTypeMeta, FewSparseDataset, FewSparseDatasetKwargs],
    ABC,
):
    @abstractmethod
    def forward(
        self,
        supp_img: Tensor,
        supp_msk: Tensor,
        qry_img: Tensor,
        loss_only: bool = False,
    ) -> tuple[Tensor, Tensor]:
        pass

    def make_dataloader(self, datasets: list[FewSparseDataset]):
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=None,
            shuffle=datasets[0].mode == "train",
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.device.type != "cpu",
        )

    def make_indices_to_save(
        self, datasets: list[FewSparseDataset], sample_size: int | None
    ) -> list[list[int]]:
        batch_size = min(ds.query_batch_size for ds in datasets)
        return make_batch_sample_indices(
            sum(ds.num_iterations for ds in datasets) * batch_size,
            sample_size or 0,
            batch_size,
        )

    def make_input_example(self) -> tuple[Any, ...]:
        batch_size = self.config["data"]["batch_size"]
        num_channels = self.config["data"]["num_channels"]
        num_classes = self.config["data"]["num_classes"]
        resize_to = self.config["data"]["resize_to"]
        supp_img_example = torch.rand(batch_size, num_channels, *resize_to)
        supp_msk_example = torch.randint(-1, num_classes, (batch_size, 1, *resize_to))
        qry_img_example = torch.rand(batch_size, num_channels, *resize_to)
        return (supp_img_example, supp_msk_example, qry_img_example)

    def training_step(self, batch: FewSparseDataTuple, batch_idx: int):
        last_epoch = self.current_epoch == self.config["learn"]["num_epochs"] - 1

        support, query, dataset_name = batch
        pred, loss = self(
            support.images, support.masks, query.images, loss_only=not last_epoch
        )

        self.training_step_losses.append(loss.item())

        self.log_to_table_metrics(
            "TR",
            batch_idx,
            loss,
            support,
        )

        if last_epoch:
            self.log_to_wandb_preds(
                "TR",
                batch_idx,
                query.masks,
                pred,
                query.file_names,
                dataset_name,
            )

        return loss

    def validation_step(self, batch: FewSparseDataTuple, batch_idx: int):
        support, query, dataset_name = batch
        pred, loss = self(support.images, support.masks, query.images)

        self.validation_step_losses.append(loss.item())
        score = self.metric(pred, query.masks)
        score = self.metric.prepare_for_log(score)

        if self.trainer.sanity_checking:
            return loss

        self.log_to_table_metrics("VL", batch_idx, loss, support, score=score)

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            self.log_to_wandb_preds(
                "VL",
                batch_idx,
                query.masks,
                pred,
                query.file_names,
                dataset_name,
            )

        return loss

    def test_step(self, batch: FewSparseDataTuple, batch_idx: int):
        support, query, dataset_name = batch
        pred, loss = self(support.images, support.masks, query.images)

        self.test_step_losses.append(loss.item())
        score = self.metric(pred, query.masks)
        score = self.metric.prepare_for_log(score)

        self.log_to_table_metrics("TS", batch_idx, loss, support, score=score, epoch=0)

        self.log_to_wandb_preds(
            "TS",
            batch_idx,
            query.masks,
            pred,
            query.file_names,
            dataset_name,
        )

        return loss

    def log_to_table_metrics(
        self,
        type: Literal["TR", "VL", "TS"],
        batch_idx: int,
        loss: Tensor,
        support: SupportDataTuple,
        score: list[tuple[str, Any]] | None = None,
        epoch: int | None = None,
    ):
        self.log_table(
            [
                ("type", type),
                ("epoch", epoch if epoch is not None else self.current_epoch),
                ("batch", batch_idx),
                ("shot", len(support.file_names)),
                ("sparsity_mode", support.sparsity_mode),
                ("sparsity_value", support.sparsity_value),
                ("loss", loss.item()),
            ]
            + (
                score
                if score is not None
                else [(key, None) for key in sorted(self.metric.additional_params())]
            ),
            "metrics",
        )
