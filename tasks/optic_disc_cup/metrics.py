import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.segmentation import hausdorff_distance

from learners.distance_metrics import DistanceMetrics
from learners.metrics import BaseMetric


def split_disc_cup(
    inputs: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if inputs.is_floating_point():
        inputs = inputs.argmax(dim=1)
    disc_targets = targets != 0
    disc_inputs = inputs != 0
    cup_targets = targets == 2
    cup_inputs = inputs == 2
    return disc_inputs, disc_targets, cup_inputs, cup_targets


class DiscCupIoU(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("iou_cup", default=torch.tensor(0), dist_reduce_fx="mean")

    @staticmethod
    def measure(inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        d_inp, d_tgt, c_inp, c_tgt = split_disc_cup(inputs, targets)
        iou_disc = binary_jaccard_index(d_inp, d_tgt)
        iou_cup = binary_jaccard_index(c_inp, c_tgt)
        return {"iou_disc": iou_disc, "iou_cup": iou_cup}


class DiscCupIoUHausdorff(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("iou_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("iou_cup", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("hausdorff_disc", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("hausdorff_cup", default=torch.tensor(0), dist_reduce_fx="mean")

    @staticmethod
    def measure(inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        d_inp, d_tgt, c_inp, c_tgt = split_disc_cup(inputs, targets)
        iou_disc = binary_jaccard_index(d_inp, d_tgt)
        iou_cup = binary_jaccard_index(c_inp, c_tgt)
        hausdorff_disc = hausdorff_distance(
            d_inp.unsqueeze(1),
            d_tgt.unsqueeze(1),
            num_classes=2,
            include_background=False,
        ).mean()
        hausdorff_cup = hausdorff_distance(
            c_inp.unsqueeze(1),
            c_tgt.unsqueeze(1),
            num_classes=2,
            include_background=False,
        ).mean()
        return {
            "iou_disc": iou_disc,
            "iou_cup": iou_cup,
            "hausdorff_disc": hausdorff_disc,
            "hausdorff_cup": hausdorff_cup,
        }


class DiscCupIoUDistance(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ["iou", "assd", "boundary_iou", "hd", "hd_perc", "masd", "nsd"]:
            self.add_state(
                f"{name}_disc", default=torch.tensor(0), dist_reduce_fx="mean"
            )
            self.add_state(
                f"{name}_cup", default=torch.tensor(0), dist_reduce_fx="mean"
            )

    @staticmethod
    def measure_distances(
        inputs: Tensor, targets: Tensor, suffix: str
    ) -> dict[str, Tensor]:
        device = inputs.device
        measures = ["assd", "boundary_iou", "hd", "hd_perc", "masd", "nsd"]

        inputs = inputs.cpu().numpy().astype(float)
        targets = targets.cpu().numpy().astype(float)
        assert inputs.shape == targets.shape

        all_distances = {k: [] for k in measures}
        batch_size = inputs.shape[0]
        for b in range(batch_size):
            inp = inputs[b]
            tgt = targets[b]
            metric = DistanceMetrics(inp, tgt, measures=measures)
            distances = metric.to_dict_meas()
            for k, v in distances.items():
                all_distances[k].append(v)
        averaged_distances = {
            k + suffix: torch.tensor(np.mean(v)).to(device)
            for k, v in all_distances.items()
        }
        return averaged_distances

    @staticmethod
    def measure(inputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        d_inp, d_tgt, c_inp, c_tgt = split_disc_cup(inputs, targets)
        iou_disc = binary_jaccard_index(d_inp, d_tgt)
        iou_cup = binary_jaccard_index(c_inp, c_tgt)
        iou_scores = {"iou_disc": iou_disc, "iou_cup": iou_cup}

        od_distances = DiscCupIoUDistance.measure_distances(d_inp, d_tgt, "_disc")
        oc_distances = DiscCupIoUDistance.measure_distances(c_inp, c_tgt, "_cup")

        return {**iou_scores, **od_distances, **oc_distances}
