import numpy as np
from numpy.typing import NDArray
from sklearn import metrics


def calc_disc_cup_iou(
    labels: list[NDArray], preds: list[NDArray]
) -> tuple[dict, str, str]:
    labels_od = [(label != 0).astype(label.dtype) for label in labels]
    labels_oc = [(label == 2).astype(label.dtype) for label in labels]
    preds_od = [(pred != 0).astype(pred.dtype) for pred in preds]
    preds_oc = [(pred == 2).astype(pred.dtype) for pred in preds]

    # Converting to numpy for computing metrics.
    labels_od_np = np.asarray(labels_od).ravel()
    labels_oc_np = np.asarray(labels_oc).ravel()
    preds_od_np = np.asarray(preds_od).ravel()
    preds_oc_np = np.asarray(preds_oc).ravel()

    # Computing metrics.
    iou_od = metrics.jaccard_score(labels_od_np, preds_od_np)
    iou_oc = metrics.jaccard_score(labels_oc_np, preds_oc_np)

    score_text = "Disc = %.2f | Cup = %.2f" % (iou_od * 100, iou_oc * 100)
    name = "IoU score"

    return {"iou_od": iou_od, "iou_oc": iou_oc}, score_text, name


def iou_to_dice(iou: float) -> float:
    return (2 * iou) / (1 + iou)
