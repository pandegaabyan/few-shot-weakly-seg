import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_jaccard_index

from learners.metrics import CustomMetric


class DiscCupIoU(CustomMetric):
    def __init__(self, **kwargs):
        super(CustomMetric, self).__init__(**kwargs)
        self.add_state("disc_iou", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("cup_iou", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, inputs: Tensor, targets: Tensor):
        inputs = inputs.argmax(dim=1)
        disc_targets = targets != 0
        disc_inputs = inputs != 0
        cup_targets = targets == 2
        cup_inputs = inputs == 2
        self.disc_iou = binary_jaccard_index(disc_inputs, disc_targets)
        self.cup_iou = binary_jaccard_index(cup_inputs, cup_targets)

    def compute(self) -> dict[str, Tensor]:
        return {"disc_iou": self.disc_iou, "cup_iou": self.cup_iou}

    def score_summary(self) -> float:
        return sum([self.disc_iou.item(), self.cup_iou.item()]) / 2

    def additional_params(self) -> dict:
        return {
            "disc_iou": "mean",
            "cup_iou": "mean",
        }
