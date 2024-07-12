from abc import ABC, abstractmethod
from typing import Any, Generic, Literal

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self, supp_image: Tensor, supp_mask: Tensor, qry_image: Tensor
    ) -> Tensor:
        pass

    @abstractmethod
    def training_process(
        self, batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def evaluation_process(
        self, type: Literal["VL", "TS"], batch: FewSparseDataTuple, batch_idx: int
    ) -> tuple[Tensor, Tensor, list[tuple[str, float]]]:
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
        self, datasets: list[FewSparseDataset], sample_size: int
    ) -> list[list[int]]:
        batch_size = min(ds.query_batch_size for ds in datasets)
        return make_batch_sample_indices(
            sum(ds.num_iterations for ds in datasets) * batch_size,
            sample_size,
            batch_size,
        )

    def make_input_example(self) -> tuple[Any, ...]:
        batch_size = self.config["data"]["batch_size"]
        num_channels = self.config["data"]["num_channels"]
        num_classes = self.config["data"]["num_classes"]
        resize_to = self.config["data"]["resize_to"]
        supp_image_example = torch.rand(batch_size, num_channels, *resize_to)
        supp_mask_example = torch.randint(-1, num_classes, (batch_size, *resize_to))
        qry_image_example = torch.rand(batch_size, num_channels, *resize_to)
        return (supp_image_example, supp_mask_example, qry_image_example)

    def training_step(self, batch: FewSparseDataTuple, batch_idx: int):
        support, query, dataset_name = batch
        pred, loss = self.training_process(batch, batch_idx)
        self.training_step_losses.append(loss.item())

        self.log_to_table_metrics(
            "TR",
            batch_idx,
            loss,
            support,
        )

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            self.wandb_log_preds(
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
        pred, loss, score = self.evaluation_process("VL", batch, batch_idx)
        self.validation_step_losses.append(loss.item())

        if self.trainer.sanity_checking:
            return loss

        self.log_to_table_metrics("VL", batch_idx, loss, support, score=score)

        if self.current_epoch == self.config["learn"]["num_epochs"] - 1:
            self.wandb_log_preds(
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
        pred, loss, score = self.evaluation_process("TS", batch, batch_idx)
        self.test_step_losses.append(loss.item())

        self.log_to_table_metrics("TS", batch_idx, loss, support, score=score, epoch=0)

        self.wandb_log_preds(
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

    def split_tensors(
        self, tensors: list[Tensor], batch_size: int = -1
    ) -> list[tuple[Tensor, ...]]:
        if batch_size == -1:
            batch_size = self.config["data"]["batch_size"]
        return [t.split(batch_size) for t in tensors]

    def manual_optimizer_step(self):
        opt_list = self.get_optimizer_list()
        for opt in opt_list:
            opt.step()

    def manual_scheduler_step(self, metrics: float | int | Tensor | None = None):
        sched_list = self.get_scheduler_list()
        for sched in sched_list:
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(metrics)
            else:
                sched.step()
