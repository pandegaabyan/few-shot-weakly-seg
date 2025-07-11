import csv
import io
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning.profilers import (
    AdvancedProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
)

from config.config_type import ProfilerType
from config.constants import FILENAMES

_TABLE_ROW = Tuple[str, float, float, int, float, float]
_TABLE_DATA = List[_TABLE_ROW]


class CustomSimpleProfiler(Profiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        percentage_limit: float = 0,
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations: Dict = defaultdict(list)
        self.start_time = time.monotonic()
        self.percentage_limit = percentage_limit

    def _prepare_filename(
        self,
        action_name: Optional[str] = None,
        extension: str = ".txt",
        split_token: str = "-",
    ) -> str:
        return super()._prepare_filename(action_name, ".csv", "_")

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def summary(self) -> str:
        total_duration = time.monotonic() - self.start_time
        report: _TABLE_DATA = []
        total_calls = 0

        for a, d in self.recorded_durations.items():
            len_d = len(d)
            if len_d == 0:
                continue
            total_calls += len_d
            d_tensor = torch.tensor(d)
            sum_d = torch.sum(d_tensor).item()
            percentage_d = 100.0 * sum_d / total_duration
            if percentage_d < self.percentage_limit:
                continue
            mean_d = torch.mean(d_tensor).item()
            if len_d == 1:
                std_d = -1
            else:
                std_d = torch.std(d_tensor).item()
            report.append((a, mean_d, std_d, len_d, sum_d, percentage_d))

        report.append(("Total", -1, -1, total_calls, total_duration, 100.0))
        report.sort(key=lambda x: x[5], reverse=True)

        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            ["Action", "Mean (s)", "Std (s)", "Num Calls", "Sum (s)", "Percentage (%)"]
        )
        writer.writerows(report)

        return output.getvalue()


def resolve_profiler(profiler_type: ProfilerType, run_path: str) -> Profiler | None:
    dirpath = os.path.join(FILENAMES["log_folder"], run_path)
    filename = FILENAMES["profile"]
    if profiler_type is None:
        return None
    elif profiler_type == "simple":
        return SimpleProfiler(dirpath=dirpath, filename=filename)
    elif profiler_type == "advanced":
        return AdvancedProfiler(dirpath=dirpath, filename=filename)
    elif profiler_type == "pytorch":
        return PyTorchProfiler(dirpath=dirpath, filename=filename)
    elif profiler_type.startswith("custom"):
        percentage_limit = float(profiler_type.split("-")[-1])
        return CustomSimpleProfiler(
            dirpath=dirpath, filename=filename, percentage_limit=percentage_limit
        )
    else:
        return None
