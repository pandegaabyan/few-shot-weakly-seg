import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_jaccard_index

from learners.metrics import CustomMetric


class DiscCupIoU(CustomMetric):
    def __init__(self, **kwargs):
        super(CustomMetric, self).__init__(**kwargs)
        self.add_state("disc_iou", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("cup_iou", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, inputs: Tensor, targets: Tensor):
        inputs = inputs.argmax(dim=1)
        disc_targets = targets != 0
        disc_inputs = inputs != 0
        cup_targets = targets == 2
        cup_inputs = inputs == 2
        self.disc_iou = binary_jaccard_index(disc_inputs, disc_targets)
        self.cup_iou = binary_jaccard_index(cup_inputs, cup_targets)

    def compute(self) -> dict[str, Tensor]:
        return {"disc_iou": self.disc_iou, "cup_iou": self.cup_iou}

    def score_summary(self) -> float:
        return sum([self.disc_iou.item(), self.cup_iou.item()]) / 2

    def additional_params(self) -> dict:
        return {
            "disc_iou": "mean",
            "cup_iou": "mean",
        }


# def calc_disc_cup_iou(
#     labels: list[NDArray], preds: list[NDArray]
# ) -> tuple[dict, str, str]:
#     labels_od = [np.not_equal(label, 0).astype(label.dtype) for label in labels]
#     labels_oc = [np.equal(label, 2).astype(label.dtype) for label in labels]
#     preds_od = [np.not_equal(pred, 0).astype(pred.dtype) for pred in preds]
#     preds_oc = [np.equal(pred, 2).astype(pred.dtype) for pred in preds]

#     # Converting to numpy for computing metrics.
#     labels_od_np = np.concatenate(labels_od, axis=0).ravel()
#     labels_oc_np = np.concatenate(labels_oc, axis=0).ravel()
#     preds_od_np = np.concatenate(preds_od, axis=0).ravel()
#     preds_oc_np = np.concatenate(preds_oc, axis=0).ravel()

#     # Computing metrics.
#     iou_od = metrics.jaccard_score(labels_od_np, preds_od_np)
#     iou_oc = metrics.jaccard_score(labels_oc_np, preds_oc_np)

#     score_text = "Disc = %.2f | Cup = %.2f" % (float(iou_od) * 100, float(iou_oc) * 100)
#     name = "IoU score"

#     return {"iou_od": iou_od, "iou_oc": iou_oc}, score_text, name
