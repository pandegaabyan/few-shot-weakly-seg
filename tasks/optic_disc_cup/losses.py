import torch
from torch import Tensor
from torch.nn import functional

from learners.losses import CustomLoss


class DiscCupLoss(CustomLoss):
    def __init__(self, loss_type: str, ignored_index: int = -1):
        super(DiscCupLoss, self).__init__()
        if loss_type not in ['ce', 'bce', 'iou', 'iou_bce']:
            raise ValueError(f'Invalid loss type: {loss_type}')
        self.loss_type = loss_type
        self.ignored_index = ignored_index

    def forward(self, inputs: Tensor, targets: Tensor):
        if self.loss_type == 'ce':
            return functional.cross_entropy(inputs, targets, ignore_index=self.ignored_index)

        od_input, oc_input, od_target, oc_target = self.process_input_and_target(inputs, targets)

        if self.loss_type == 'iou':
            od_loss = self.iou_loss(od_input, od_target)
            oc_loss = self.iou_loss(oc_input, oc_target)
        elif self.loss_type == 'iou_bce':
            od_loss = self.iou_bce_loss(od_input, od_target)
            oc_loss = self.iou_bce_loss(oc_input, oc_target)
        else:
            od_loss = functional.binary_cross_entropy(od_input, od_target)
            oc_loss = functional.binary_cross_entropy(oc_input, oc_target)

        return od_loss + oc_loss

    def process_input_and_target(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        od_input = functional.softmax(inputs, 1)[:, 1:3].sum(1)
        oc_input = functional.softmax(inputs, 1)[:, 2]

        od_target = torch.clone(targets)
        od_target[od_target == 2] = 1
        oc_target = torch.clone(targets)
        oc_target[oc_target == 1] = 0
        oc_target[oc_target == 2] = 1

        od_input_filtered = od_input[od_target != self.ignored_index]
        oc_input_filtered = oc_input[oc_target != self.ignored_index]
        od_target_filtered = od_target[od_target != self.ignored_index].type(torch.float32)
        oc_target_filtered = oc_target[oc_target != self.ignored_index].type(torch.float32)

        return od_input_filtered, oc_input_filtered, od_target_filtered, oc_target_filtered
