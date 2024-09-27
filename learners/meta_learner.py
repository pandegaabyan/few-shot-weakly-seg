from abc import ABC, abstractmethod
from typing import Any, Generic, Literal

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

from data.few_sparse_dataset import FewSparseDataset
from data.typings import (
    FewSparseDatasetKwargs,
    FewSparseDataTuple,
    SparsityValue,
    SupportDataTuple,
)
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
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        pass

    def make_dataloader(self, datasets: list[FewSparseDataset]):
        num_workers = self.config["data"]["num_workers"]
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=None,
            shuffle=datasets[0].mode == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.device.type != "cpu",
        )

    def make_indices_to_save(
        self, datasets: list[FewSparseDataset], sample_size: int
    ) -> list[list[int]] | None:
        batch_size = min(ds.query_batch_size for ds in datasets)
        return make_batch_sample_indices(
            sum(ds.num_iterations for ds in datasets) * batch_size,
            sample_size,
            batch_size,
            seed=self.config["learn"].get("seed"),
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
        with self.profile("training_process"):
            pred, loss = self.training_process(batch, batch_idx)
        self.training_step_losses.append(loss.item())

        self.handle_metrics(
            "TR",
            batch_idx,
            loss,
            batch.support,
        )

        self.handle_preds("TS", batch, batch_idx, pred)

        return loss

    def validation_step(self, batch: FewSparseDataTuple, batch_idx: int):
        filename = f"val {self.current_epoch}.txt"
        with open(filename, "a") as f:
            f.write(f"{batch.support.indices} {batch.query.indices}\n")

        with self.profile("evaluation_process"):
            pred, loss, score = self.evaluation_process("VL", batch, batch_idx)
        self.validation_step_losses.append(loss.item())

        if self.trainer.sanity_checking:
            return loss

        self.handle_metrics("VL", batch_idx, loss, batch.support, score=score)

        self.handle_preds("VL", batch, batch_idx, pred)

        return loss

    def test_step(self, batch: FewSparseDataTuple, batch_idx: int):
        with self.profile("evaluation_process"):
            pred, loss, score = self.evaluation_process("TS", batch, batch_idx)
        self.test_step_losses.append(loss.item())

        self.handle_metrics("TS", batch_idx, loss, batch.support, score=score, epoch=0)

        self.handle_preds("TS", batch, batch_idx, pred)

        return loss

    def encode_sparsity_value(self, value: SparsityValue) -> str:
        return value if isinstance(value, str) else str(round(value, 3))

    def handle_metrics(
        self,
        type: Literal["TR", "VL", "TS"],
        batch_idx: int,
        loss: Tensor,
        support: SupportDataTuple,
        score: dict[str, Tensor] | None = None,
        epoch: int | None = None,
    ):
        if not self.config["log"].get("table"):
            return

        if score is not None:
            score_tup = self.metric.prepare_for_log(score)
        else:
            score_tup = [(key, None) for key in sorted(self.metric.metrics)]

        if isinstance(support.sparsity_mode, list):
            sparsity_mode = " ".join(support.sparsity_mode)
        else:
            sparsity_mode = support.sparsity_mode

        if isinstance(support.sparsity_value, list):
            sparsity_value = " ".join(
                map(self.encode_sparsity_value, support.sparsity_value)
            )
        else:
            sparsity_value = self.encode_sparsity_value(support.sparsity_value)

        self.log_table(
            [
                ("type", type),
                ("epoch", epoch if epoch is not None else self.current_epoch),
                ("batch", batch_idx),
                ("shot", len(support.indices)),
                ("sparsity_mode", sparsity_mode),
                ("sparsity_value", sparsity_value),
                ("loss", loss.item()),
            ]
            + score_tup,
            "metrics",
        )

    def handle_preds(
        self,
        type: Literal["TR", "VL", "TS"],
        batch: FewSparseDataTuple,
        batch_idx: int,
        preds: Tensor,
    ):
        support, query, dataset = batch
        if isinstance(support.sparsity_value, list):
            sparsity_value = list(
                map(self.encode_sparsity_value, support.sparsity_value)
            )
        else:
            sparsity_value = support.sparsity_value
        self.wandb_handle_preds(
            type,
            batch_idx,
            preds,
            query.indices,
            dataset,
            shot=len(support.indices),
            sparsity_mode=support.sparsity_mode,
            sparsity_value=sparsity_value,
        )

    def split_tensors(
        self, tensors: list[Tensor], batch_size: int = -1
    ) -> list[tuple[Tensor, ...]]:
        if batch_size == -1:
            batch_size = self.config["data"]["batch_size"]
        return [t.split(batch_size) for t in tensors]

    def optimizer_state_dicts(self) -> list[dict[str, Any]]:
        return [opt.optimizer.state_dict() for opt in self.get_optimizer_list()]

    def load_optimizer_state_dicts(self, state_dicts: list[dict[str, Any]]):
        opt_list = self.get_optimizer_list()
        for opt, state_dict in zip(opt_list, state_dicts):
            opt.optimizer.load_state_dict(state_dict)

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
