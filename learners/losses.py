import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomLoss(nn.Module):
    def __init__(
        self,
        mode: str = "ce",
        ignored_index: int = -1,
        ce_weights: Tensor | None = None,
    ):
        super().__init__()
        if mode not in self.valid_modes():
            raise ValueError(f"Invalid loss type: {mode}")
        self.mode = mode
        self.ignored_index = ignored_index
        self.ce_weights = Tensor(ce_weights) if ce_weights is not None else None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.ce_weights is not None and self.ce_weights.device != inputs.device:
            self.ce_weights = self.ce_weights.to(inputs.device)

        if self.mode == "bce":
            return self.bce_loss(
                inputs, targets, ignore_index=self.ignored_index, weight=self.ce_weights
            )

        if self.mode == "biou":
            return self.biou_loss(inputs, targets, ignore_index=self.ignored_index)

        return self.ce_loss(
            inputs, targets, ignore_index=self.ignored_index, weight=self.ce_weights
        )

    def valid_modes(self) -> list[str]:
        return ["ce", "bce", "biou"]

    @staticmethod
    def ce_loss(
        inputs: Tensor,
        targets: Tensor,
        ignore_index: int = -1,
        weight: Tensor | None = None,
    ):
        return F.cross_entropy(
            inputs, targets, ignore_index=ignore_index, weight=weight
        )

    @staticmethod
    def bce_loss(
        inputs: Tensor,
        targets: Tensor,
        ignore_index: int = -1,
        weight: Tensor | None = None,
    ):
        new_inputs, new_targets = CustomLoss.validate_and_filter_binary(
            inputs, targets, ignore_index
        )
        return F.binary_cross_entropy_with_logits(
            new_inputs, new_targets.float(), pos_weight=weight
        )

    @staticmethod
    def biou_loss(
        inputs: Tensor, targets: Tensor, ignore_index: int = -1, smooth: int = 1
    ):
        new_inputs, new_targets = CustomLoss.validate_and_filter_binary(
            inputs, targets, ignore_index
        )
        new_inputs = F.sigmoid(new_inputs)

        intersection = (new_inputs * new_targets).sum()
        union = (new_inputs + new_targets).sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou

    @staticmethod
    def validate_and_filter_binary(
        inputs: Tensor, targets: Tensor, ignore_index: int = -1
    ):
        in_size = inputs.size()
        tg_size = targets.size()
        if in_size != tg_size:
            if in_size[1] == 1 and in_size[-2:] == tg_size[-2:]:
                inputs = inputs[:, 0]
            else:
                raise ValueError(
                    f"Inputs and targets must have compatible shape, got {in_size} and {tg_size}"
                )

        filtered_inputs = inputs[targets != ignore_index]
        filtered_targets = targets[targets != ignore_index]
        return filtered_inputs, filtered_targets

    # def set_ce_weights_from_target(
    #     self, targets: Tensor, num_classes: int, device: device
    # ):
    #     weights = torch.Tensor([torch.sum(targets == c) for c in range(num_classes)])
    #     if torch.any(weights == 0):
    #         weights = torch.Tensor([1 for _ in range(num_classes)])
    #     else:
    #         weights = weights / weights.max()
    #     weights = weights.to(device=device)
    #     self.ce_weights = weights
