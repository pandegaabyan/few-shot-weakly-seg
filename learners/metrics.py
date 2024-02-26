from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_jaccard_index


class CustomMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, inputs: Tensor, targets: Tensor):
        self.iou = multiclass_jaccard_index(inputs, targets, targets.unique().numel())

    def compute(self) -> dict[str, Tensor]:
        return {"iou": self.iou}

    def score_summary(self) -> float:
        return self.iou.item()

    def additional_params(self) -> dict[str, Any]:
        return {
            "iou": "mean",
        }

    def params(self) -> dict:
        return {
            "training": self.training,
            **self.additional_params(),
        }

    @staticmethod
    def prepare_for_log(score: dict[str, Tensor]) -> list[tuple[str, float]]:
        return [(k, round(v.item(), 4)) for k, v in sorted(score.items())]
