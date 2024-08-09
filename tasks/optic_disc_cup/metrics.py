from typing import Any

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_jaccard_index

from learners.metrics import BaseMetric


class DiscCupIoU(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("iou_cup", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, inputs: Tensor, targets: Tensor):
        if inputs.is_floating_point():
            inputs = inputs.argmax(dim=1)
        disc_targets = targets != 0
        disc_inputs = inputs != 0
        cup_targets = targets == 2
        cup_inputs = inputs == 2
        self.iou_disc = binary_jaccard_index(disc_inputs, disc_targets)
        self.iou_cup = binary_jaccard_index(cup_inputs, cup_targets)

    def compute(self) -> dict[str, Tensor]:
        return {"iou_disc": self.iou_disc, "iou_cup": self.iou_cup}

    def score_summary(self) -> float:
        return sum([self.iou_disc.item(), self.iou_cup.item()]) / 2

    def additional_params(self) -> dict[str, Any]:
        return {
            "iou_disc": "mean",
            "iou_cup": "mean",
        }
