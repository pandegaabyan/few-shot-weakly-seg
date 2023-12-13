import numpy as np
import sklearn
from numpy.typing import NDArray


def calc_print_disc_cup_iou(labels: list[NDArray], preds: list[NDArray], message: str):
    iou_od, iou_oc = calc_disc_cup_iou(labels, preds)

    final_message = 'IoU score' if message == '' else 'IoU score ' + message
    print('%s: Disc = %.2f | Cup = %.2f' % (
        final_message, iou_od * 100, iou_oc * 100))


def calc_disc_cup_iou(labels: list[NDArray], preds: list[NDArray]) -> tuple[float, float]:
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
    iou_od = sklearn.metrics.jaccard_score(labels_od_np, preds_od_np)
    iou_oc = sklearn.metrics.jaccard_score(labels_oc_np, preds_oc_np)

    return iou_od, iou_oc


def iou_to_dice(iou: float) -> float:
    return (2*iou)/(1+iou)
