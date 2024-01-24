import torch
from torch import Tensor
from torch.nn import functional

from learners.losses import CustomLoss


class DiscCupLoss(CustomLoss):
    def __init__(self, loss_type: str, ignored_index: int = -1):
        super(DiscCupLoss, self).__init__()
        if loss_type not in ['ce', 'bce', 'mce', 'iou', 'iou_bce']:
            raise ValueError(f'Invalid loss type: {loss_type}')
        self.loss_type = loss_type
        self.ignored_index = ignored_index

    def forward(self, inputs: Tensor, targets: Tensor):
        if self.loss_type == 'ce':
            return self.ce_loss(inputs, targets, ignore_index=self.ignored_index)

        if self.loss_type == 'bce':
            od_input, oc_input, od_target, oc_target = self.split_od_oc(inputs, targets)
            od_loss = self.bce_loss(od_input, od_target, ignore_index=self.ignored_index)
            oc_loss = self.bce_loss(oc_input, oc_target, ignore_index=self.ignored_index)
            return od_loss + oc_loss

        if self.loss_type == 'mce':
            return self.mce_loss(inputs, targets,
                                 ignore_index=self.ignored_index, weight=self.mce_weights)

        if self.loss_type == 'iou':
            od_input, oc_input, od_target, oc_target = self.process_input_and_target(inputs, targets)
            od_loss = self.iou_loss(od_input, od_target, self.iou_smooth)
            oc_loss = self.iou_loss(oc_input, oc_target, self.iou_smooth)
            return od_loss + oc_loss

        if self.loss_type == 'iou_bce':
            od_input, oc_input, od_target, oc_target = self.process_input_and_target(inputs, targets)
            od_loss_iou = self.iou_loss(od_input, od_target, self.iou_smooth)
            oc_loss_iou = self.iou_loss(oc_input, oc_target, self.iou_smooth)
            od_input, oc_input, od_target, oc_target = self.split_od_oc(inputs, targets)
            od_loss_bce = self.bce_loss(od_input, od_target, ignore_index=self.ignored_index)
            oc_loss_bce = self.bce_loss(oc_input, oc_target, ignore_index=self.ignored_index)
            return od_loss_iou + oc_loss_iou + od_loss_bce + oc_loss_bce

        return 0

    def params(self) -> dict:
        return {'loss_type': self.loss_type, 'ignored_index': self.ignored_index}

    @staticmethod
    def split_od_oc(inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        od_input = inputs[:, 1:3].sum(1)
        oc_input = inputs[:, 2]

        od_target = torch.clone(targets)
        od_target[od_target == 2] = 1
        oc_target = torch.clone(targets)
        oc_target[oc_target == 1] = 0
        oc_target[oc_target == 2] = 1

        return od_input, oc_input, od_target, oc_target

    def process_input_and_target(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        soft_inputs = functional.softmax(inputs, 1)

        od_input, oc_input, od_target, oc_target = self.split_od_oc(soft_inputs, targets)

        od_input_filtered = od_input[od_target != self.ignored_index]
        oc_input_filtered = oc_input[oc_target != self.ignored_index]
        od_target_filtered = od_target[od_target != self.ignored_index]
        oc_target_filtered = oc_target[oc_target != self.ignored_index]

        return od_input_filtered, oc_input_filtered, od_target_filtered, oc_target_filtered
