import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_jaccard_index

from learners.metrics import BaseMetric


class DiscCupIoU(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("iou_cup", default=torch.tensor(0), dist_reduce_fx="mean")

    def measure(self, inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        if inputs.is_floating_point():
            inputs = inputs.argmax(dim=1)
        disc_targets = targets != 0
        disc_inputs = inputs != 0
        cup_targets = targets == 2
        cup_inputs = inputs == 2
        iou_disc = binary_jaccard_index(disc_inputs, disc_targets)
        iou_cup = binary_jaccard_index(cup_inputs, cup_targets)
        return {"iou_disc": iou_disc, "iou_cup": iou_cup}
