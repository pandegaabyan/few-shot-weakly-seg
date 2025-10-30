import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.segmentation import hausdorff_distance

from learners.metrics import BaseMetric


class DiscCupMetric(BaseMetric):
    @staticmethod
    def split_disc_cup(
        inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if inputs.is_floating_point():
            inputs = inputs.argmax(dim=1)
        disc_targets = targets != 0
        disc_inputs = inputs != 0
        cup_targets = targets == 2
        cup_inputs = inputs == 2
        return disc_inputs, disc_targets, cup_inputs, cup_targets


class DiscCupIoU(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("iou_cup", default=torch.tensor(0), dist_reduce_fx="mean")

    @staticmethod
    def measure(inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        d_inp, d_tgt, c_inp, c_tgt = DiscCupMetric.split_disc_cup(inputs, targets)
        iou_disc = binary_jaccard_index(d_inp, d_tgt)
        iou_cup = binary_jaccard_index(c_inp, c_tgt)
        return {"iou_disc": iou_disc, "iou_cup": iou_cup}


class DiscCupIoUHausdorff(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("iou_cup", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("hausdorff_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("hausdorff_cup", default=torch.tensor(0), dist_reduce_fx="mean")

    @staticmethod
    def measure(inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        d_inp, d_tgt, c_inp, c_tgt = DiscCupMetric.split_disc_cup(inputs, targets)
        iou_disc = binary_jaccard_index(d_inp, d_tgt)
        iou_cup = binary_jaccard_index(c_inp, c_tgt)
        hausdorff_disc = hausdorff_distance(
            d_inp.unsqueeze(1),
            d_tgt.unsqueeze(1),
            num_classes=2,
            include_background=False,
        )
        hausdorff_cup = hausdorff_distance(
            c_inp.unsqueeze(1),
            c_tgt.unsqueeze(1),
            num_classes=2,
            include_background=False,
        )
        return {
            "iou_disc": iou_disc,
            "iou_cup": iou_cup,
            "hausdorff_disc": hausdorff_disc,
            "hausdorff_cup": hausdorff_cup,
        }
