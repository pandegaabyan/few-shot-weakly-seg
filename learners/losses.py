from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torch.nn import functional


class CustomLoss(nn.Module, ABC):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @abstractmethod
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return functional.cross_entropy(inputs, targets, ignore_index=-1)

    @staticmethod
    def iou_loss(inputs: Tensor, targets: Tensor, smooth=1):
        intersection = (inputs * targets).sum()
        union = (inputs + targets).sum() - intersection
        iou = (intersection + smooth)/(union + smooth)
        return 1 - iou

    @staticmethod
    # TODO: currently unstable causing CUDA crash despite low memory usage
    def iou_bce_loss(inputs: Tensor, targets: Tensor, smooth=1):
        intersection = (inputs * targets).sum()
        union = (inputs + targets).sum() - intersection
        iou = (intersection + smooth)/(union + smooth)
        bce = functional.binary_cross_entropy(inputs, targets)
        iou_bce = (1 - iou) + bce
        return iou_bce
