import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_type = 'ce'
        self.ignored_index = -1

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return functional.cross_entropy(inputs, targets, ignore_index=self.ignored_index)

    def params(self) -> dict:
        return {'loss_type': self.loss_type, 'ignored_index': self.ignored_index}

    @staticmethod
    def iou_loss(inputs: Tensor, targets: Tensor, smooth: int = 1):
        intersection = (inputs * targets).sum()
        union = (inputs + targets).sum() - intersection
        iou = (intersection + smooth)/(union + smooth)
        return 1 - iou

    @staticmethod
    def bce_loss(inputs: Tensor, targets: Tensor, ignore_index: int = -1):
        expanded_inputs = torch.stack([inputs, 1 - inputs], dim=1)
        return functional.cross_entropy(expanded_inputs, targets, ignore_index=ignore_index)
