import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_type = "ce"
        self.ignored_index = -1
        self.mce_weights = None
        self.iou_smooth = 1

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.loss_type == "ce":
            return self.ce_loss(inputs, targets, ignore_index=self.ignored_index)

        if self.loss_type == "bce":
            return self.bce_loss(inputs, targets, ignore_index=self.ignored_index)

        if self.loss_type == "mce":
            return self.mce_loss(
                inputs,
                targets,
                ignore_index=self.ignored_index,
                weight=self.mce_weights,
            )

        if self.loss_type == "iou":
            return self.iou_loss(inputs, targets, smooth=self.iou_smooth)

    def params(self) -> dict:
        return {"loss_type": self.loss_type, "ignored_index": self.ignored_index}

    @staticmethod
    def ce_loss(inputs: Tensor, targets: Tensor, ignore_index: int = -1):
        return functional.cross_entropy(inputs, targets, ignore_index=ignore_index)

    @staticmethod
    def bce_loss(inputs: Tensor, targets: Tensor, ignore_index: int = -1):
        expanded_inputs = torch.stack([inputs, 1 - inputs], dim=1)
        return functional.cross_entropy(
            expanded_inputs, targets, ignore_index=ignore_index
        )

    @staticmethod
    def mce_loss(
        inputs: Tensor,
        targets: Tensor,
        ignore_index: int = -1,
        weight: Tensor | None = None,
    ):
        return functional.cross_entropy(
            inputs, targets, reduction="mean", ignore_index=ignore_index, weight=weight
        )

    @staticmethod
    def iou_loss(inputs: Tensor, targets: Tensor, smooth: int = 1):
        intersection = (inputs * targets).sum()
        union = (inputs + targets).sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou

    def set_mce_weights_from_target(
        self, targets: Tensor, num_classes: int, use_gpu: bool = False
    ):
        weights = torch.Tensor([torch.sum(targets == c) for c in range(num_classes)])  # type: ignore
        if torch.any(weights == 0):  # type: ignore
            weights = torch.Tensor([1 for _ in range(num_classes)])
        else:
            weights = weights / weights.max()
        if use_gpu:
            weights = weights.cuda()
        self.mce_weights = weights

    def set_iou_smooth(self, smooth: int = 1):
        self.iou_smooth = smooth
