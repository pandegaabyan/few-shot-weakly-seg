from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_jaccard_index


class BaseMetric(Metric, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def score_summary(self) -> float:
        pass

    @staticmethod
    def prepare_for_log(score: dict[str, Tensor]) -> list[tuple[str, float]]:
        return [(k, round(v.item(), 4)) for k, v in sorted(score.items())]


class MultiIoUMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, inputs: Tensor, targets: Tensor):
        self.iou = multiclass_jaccard_index(inputs, targets, targets.unique().numel())

    def compute(self) -> dict[str, Tensor]:
        return {"iou": self.iou}

    def set_state_info(self) -> dict[str, str | None]:
        return {
            "iou": "mean",
        }

    def score_summary(self) -> float:
        return self.iou.item()
