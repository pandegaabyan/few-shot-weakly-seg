from __future__ import absolute_import, print_function

import warnings
from functools import partial

import numpy as np
from scipy import ndimage


class CacheFunctionOutput(object):
    """
    this provides a decorator to cache function outputs
    to avoid repeating some heavy function computations
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return partial(self, obj)  # to remember func as self.func

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*args, **kw)
        return value


class MorphologyOps(object):
    """
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    """

    def __init__(self, binary_img, connectivity):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.connectivity = connectivity

    def border_map(self):
        eroded = ndimage.binary_erosion(self.binary_map)
        border = self.binary_map - eroded  # type: ignore
        return border


class DistanceMetrics(object):
    def __init__(
        self,
        pred,
        ref,
        measures=[],
        connectivity_type=1,
        pixdim=None,
        empty=False,
        dict_args={},
    ):
        self.measures_dict = {
            "assd": (self.measured_average_distance, "ASSD"),
            "boundary_iou": (self.boundary_iou, "BoundaryIoU"),
            "hd": (self.measured_hausdorff_distance, "HD"),
            "hd_perc": (self.measured_hausdorff_distance_perc, "HDPerc"),
            "masd": (self.measured_masd, "MASD"),
            "nsd": (self.normalised_surface_distance, "NSD"),
        }

        self.pred = pred
        self.ref = ref
        self.flag_empty = empty
        self.flag_empty_pred = False
        self.flag_empty_ref = False
        if np.sum(self.pred) == 0:
            self.flag_empty_pred = True
        if np.sum(self.ref) == 0:
            self.flag_empty_ref = True
        self.measures = measures if measures is not None else self.measures_dict
        self.connectivity = connectivity_type
        self.pixdim = pixdim
        self.dict_args = dict_args

    @CacheFunctionOutput
    def n_pos_ref(self):
        n_pos_ref = np.sum(self.ref)
        return n_pos_ref

    @CacheFunctionOutput
    def n_neg_ref(self):
        n_neg_ref = np.sum(1 - self.ref)
        return n_neg_ref

    @CacheFunctionOutput
    def n_pos_pred(self):
        n_pos_pred = np.sum(self.pred)
        return n_pos_pred

    @CacheFunctionOutput
    def n_neg_pred(self):
        n_neg_pred = np.sum(1 - self.pred)
        return n_neg_pred

    def boundary_iou(self):
        if "boundary_dist" in self.dict_args.keys():
            distance = self.dict_args["boundary_dist"]
        else:
            distance = 1
        if int(self.n_pos_ref()) == 0 and int(self.n_pos_pred()) == 0:
            warnings.warn(
                "Both prediction and reference empty - setting to max for boudnary ioU"
            )
            return 1
        else:
            border_ref = MorphologyOps(self.ref, self.connectivity).border_map()
            distance_border_ref = ndimage.distance_transform_edt(1 - border_ref)

            border_pred = MorphologyOps(self.pred, self.connectivity).border_map()
            distance_border_pred = ndimage.distance_transform_edt(1 - border_pred)

            if (
                distance_border_ref is None
                or distance_border_pred is None
                or isinstance(distance_border_ref, tuple)
                or isinstance(distance_border_pred, tuple)
            ):
                raise ValueError("Distance transform could not be computed.")

            lim_dbp = np.where(
                np.logical_and(distance_border_pred < distance, self.pred > 0),
                np.ones_like(border_pred),
                np.zeros_like(border_pred),
            )
            lim_dbr = np.where(
                np.logical_and(distance_border_ref < distance, self.ref > 0),
                np.ones_like(border_ref),
                np.zeros_like(border_ref),
            )

            intersect = np.sum(lim_dbp * lim_dbr)
            union = np.sum(
                np.where(
                    lim_dbp + lim_dbr > 0,
                    np.ones_like(border_ref),
                    np.zeros_like(border_pred),
                )
            )
            boundary_iou = intersect / union
            return boundary_iou

    @CacheFunctionOutput
    def border_distance(self):
        border_ref = MorphologyOps(self.ref, self.connectivity).border_map()
        border_pred = MorphologyOps(self.pred, self.connectivity).border_map()
        distance_ref = ndimage.distance_transform_edt(
            1 - border_ref, sampling=self.pixdim
        )
        distance_pred = ndimage.distance_transform_edt(
            1 - border_pred, sampling=self.pixdim
        )
        if distance_ref is None or distance_pred is None:
            raise ValueError("Distance transform could not be computed.")
        distance_border_pred = border_ref * distance_pred
        distance_border_ref = border_pred * distance_ref
        return distance_border_ref, distance_border_pred, border_ref, border_pred

    def normalised_surface_distance(self):
        if "nsd" in self.dict_args.keys():
            tau = self.dict_args["nsd"]
        else:
            warnings.warn("No value set up for NSD tolerance - default to 1")
            tau = 1
        if int(self.n_pos_pred()) == 0 and int(self.n_pos_ref()) == 0:
            warnings.warn("Both reference and prediction are empty - setting to best")
            return 1
        else:
            dist_ref, dist_pred, border_ref, border_pred = self.border_distance()
            reg_ref = np.where(
                dist_ref <= tau, np.ones_like(dist_ref), np.zeros_like(dist_ref)
            )
            reg_pred = np.where(
                dist_pred <= tau, np.ones_like(dist_pred), np.zeros_like(dist_pred)
            )
            numerator = np.sum(border_pred * reg_ref) + np.sum(border_ref * reg_pred)
            denominator = np.sum(border_ref) + np.sum(border_pred)
            return numerator / denominator

    def measured_distance(self):
        if "hd_perc" in self.dict_args.keys():
            perc = self.dict_args["hd_perc"]
        else:
            warnings.warn(
                "Percentile not specified in options for Hausdorff distance - default set to 95"
            )
            perc = 95
        if np.sum(self.pred + self.ref) == 0:
            warnings.warn("Prediction and reference empty - distances set to 0")
            return 0, 0, 0, 0
        if np.sum(self.pred) == 0 and np.sum(self.ref) > 0:
            warnings.warn(
                "Prediction empty but reference not empty - need to set to worse case in aggregation"
            )
            return np.nan, np.nan, np.nan, np.nan
        if np.sum(self.ref) == 0 and np.sum(self.pred) > 0:
            warnings.warn(
                "Prediction not empty but reference empty - non existing output - need be set to WORSE case in aggregation"
            )
            return np.nan, np.nan, np.nan, np.nan
        (
            ref_border_dist,
            pred_border_dist,
            ref_border,
            pred_border,
        ) = self.border_distance()
        average_distance = (np.sum(ref_border_dist) + np.sum(pred_border_dist)) / (
            np.sum(pred_border + ref_border)
        )
        masd = 0.5 * (
            np.sum(ref_border_dist) / np.sum(pred_border)
            + np.sum(pred_border_dist) / np.sum(ref_border)
        )

        hausdorff_distance = np.max([np.max(ref_border_dist), np.max(pred_border_dist)])

        hausdorff_distance_perc = np.max(
            [
                np.percentile(ref_border_dist[pred_border > 0], q=perc),
                np.percentile(pred_border_dist[ref_border > 0], q=perc),
            ]
        )

        return hausdorff_distance, average_distance, hausdorff_distance_perc, masd

    def measured_average_distance(self):
        assd = self.measured_distance()[1]
        return assd

    def measured_masd(self):
        masd = self.measured_distance()[3]
        return masd

    def measured_hausdorff_distance(self):
        hausdorff_distance = self.measured_distance()[0]
        return hausdorff_distance

    def measured_hausdorff_distance_perc(self):
        hausdorff_distance_perc = self.measured_distance()[2]
        return hausdorff_distance_perc

    def to_dict_meas(self, fmt="{:.4f}"):
        result_dict = {}
        for key in self.measures:
            if len(self.measures_dict[key]) == 2:
                result = self.measures_dict[key][0]()
            else:
                result = self.measures_dict[key][0](self.measures_dict[key][2])
            result_dict[key] = result
        return result_dict
