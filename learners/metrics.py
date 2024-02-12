import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_jaccard_index


class CustomMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, inputs: Tensor, targets: Tensor):
        self.iou = multiclass_jaccard_index(inputs, targets, targets.unique().numel())

    def compute(self) -> dict[str, Tensor]:
        return {"iou": self.iou}

    def score_summary(self) -> float:
        return self.iou.item()

    def additional_params(self) -> dict:
        return {
            "iou": "mean",
        }

    def params(self) -> dict:
        return {
            "training": self.training,
            # "compute_on_cpu": self.compute_on_cpu,
            # "dist_sync_on_step": self.dist_sync_on_step,
            # "sync_on_compute": self.sync_on_compute,
            # "compute_with_cache": self.compute_with_cache,
            **self.additional_params(),
        }

    @staticmethod
    def prepare_for_log(score: dict[str, Tensor]) -> list[tuple[str, float]]:
        return [(k, round(v.item(), 4)) for k, v in sorted(score.items())]
