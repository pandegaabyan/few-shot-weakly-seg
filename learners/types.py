from typing import Callable

from numpy.typing import NDArray
from torch import nn, optim

NeuralNetworks = nn.Module | dict[str, nn.Module]

CalcMetrics = Callable[[list[NDArray], list[NDArray]], tuple[dict, str, str]]

Optimizer = optim.Optimizer

Scheduler = optim.lr_scheduler.LRScheduler
