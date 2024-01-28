import numpy as np
from numpy.typing import NDArray
from sklearn import metrics


def calc_disc_cup_iou(
    labels: list[NDArray], preds: list[NDArray]
) -> tuple[dict, str, str]:
    labels_od = [np.not_equal(label, 0).astype(label.dtype) for label in labels]
    labels_oc = [np.equal(label, 2).astype(label.dtype) for label in labels]
    preds_od = [np.not_equal(pred, 0).astype(pred.dtype) for pred in preds]
    preds_oc = [np.equal(pred, 2).astype(pred.dtype) for pred in preds]

    # Converting to numpy for computing metrics.
    labels_od_np = np.concatenate(labels_od, axis=0).ravel()
    labels_oc_np = np.concatenate(labels_oc, axis=0).ravel()
    preds_od_np = np.concatenate(preds_od, axis=0).ravel()
    preds_oc_np = np.concatenate(preds_oc, axis=0).ravel()

    # Computing metrics.
    iou_od = metrics.jaccard_score(labels_od_np, preds_od_np)
    iou_oc = metrics.jaccard_score(labels_oc_np, preds_oc_np)

    score_text = "Disc = %.2f | Cup = %.2f" % (float(iou_od) * 100, float(iou_oc) * 100)
    name = "IoU score"

    return {"iou_od": iou_od, "iou_oc": iou_oc}, score_text, name


def iou_to_dice(iou: float) -> float:
    return (2 * iou) / (1 + iou)
