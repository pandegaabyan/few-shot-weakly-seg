from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_jaccard_index

from utils.utils import mean


class BaseMetric(Metric, ABC):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metrics: set[str] = set()

    def add_state(
        self,
        name: str,
        default: Tensor,
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        self.metrics.add(name)
        return super().add_state(name, default, dist_reduce_fx, persistent)

    @abstractmethod
    def measure(self, inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        pass

    def update(self, inputs: Tensor, targets: Tensor):
        scores = self.measure(inputs, targets)
        for k, v in scores.items():
            setattr(self, k, v)

    def compute(self) -> dict[str, Tensor]:
        return {k: getattr(self, k) for k in self.metrics}

    def score_summary(self) -> float:
        return mean([getattr(self, k).item() for k in self.metrics])

    @staticmethod
    def prepare_for_log(score: dict[str, Tensor]) -> list[tuple[str, float]]:
        return [(k, round(v.item(), 4)) for k, v in sorted(score.items())]


class MultiIoUMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou", default=torch.tensor(0), dist_reduce_fx="mean")

    def measure(self, inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        return {
            "iou": multiclass_jaccard_index(inputs, targets, targets.unique().numel())
        }
