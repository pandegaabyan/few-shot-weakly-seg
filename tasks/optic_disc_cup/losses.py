import torch
from torch import Tensor

from learners.losses import CustomLoss


class DiscCupLoss(CustomLoss):
    def forward(self, inputs: Tensor, targets: Tensor):
        if self.mode in ["bce", "iou", "iou_bce"]:
            od_input, oc_input, od_target, oc_target = self.split_od_oc(inputs, targets)

        if self.mode == "bce":
            od_loss = self.bce_loss(
                od_input, od_target, ignore_index=self.ignored_index
            )
            oc_loss = self.bce_loss(
                oc_input, oc_target, ignore_index=self.ignored_index
            )
            return od_loss + oc_loss

        if self.mode == "iou":
            od_loss = self.biou_loss(
                od_input, od_target, ignore_index=self.ignored_index
            )
            oc_loss = self.biou_loss(
                oc_input, oc_target, ignore_index=self.ignored_index
            )
            return od_loss + oc_loss

        if self.mode == "iou_bce":
            od_bce_loss = self.bce_loss(
                od_input, od_target, ignore_index=self.ignored_index
            )
            oc_bce_loss = self.bce_loss(
                oc_input, oc_target, ignore_index=self.ignored_index
            )
            od_iou_loss = self.biou_loss(
                od_input, od_target, ignore_index=self.ignored_index
            )
            oc_iou_loss = self.biou_loss(
                oc_input, oc_target, ignore_index=self.ignored_index
            )
            return od_bce_loss + oc_bce_loss + od_iou_loss + oc_iou_loss

        return self.ce_loss(
            inputs, targets, ignore_index=self.ignored_index, weight=self.ce_weights
        )

    def valid_modes(self) -> list[str]:
        return ["ce", "bce", "iou", "iou_bce"]

    @staticmethod
    def split_od_oc(
        inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        od_input = inputs[:, 1:3].sum(1)
        oc_input = inputs[:, 2]

        od_target = torch.clone(targets)
        od_target[od_target == 2] = 1
        oc_target = torch.clone(targets)
        oc_target[oc_target == 1] = 0
        oc_target[oc_target == 2] = 1

        return od_input, oc_input, od_target, oc_target
