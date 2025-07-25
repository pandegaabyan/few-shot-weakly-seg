import random
from abc import ABC, abstractmethod
from math import floor

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import data as skdata
from skimage import measure, morphology, segmentation
from typing_extensions import Unpack

from data.base_dataset import BaseDataset
from data.typings import (
    DatasetModes,
    FewSparseDatasetKwargs,
    FewSparseDataTuple,
    QueryDataTuple,
    SparsityMode,
    SparsityTuple,
    SparsityValue,
    SupportDataTuple,
)


class FewSparseDataset(BaseDataset, ABC):
    def __init__(
        self,
        mode: DatasetModes,
        num_classes: int,
        resize_to: tuple[int, int],
        **kwargs: Unpack[FewSparseDatasetKwargs],
    ):
        super().__init__(mode, num_classes, resize_to, **kwargs)

        self.shot_options = kwargs.get("shot_options", "all")
        self.sparsity_options = kwargs.get("sparsity_options") or [("random", "random")]
        self.sparsity_params = kwargs.get("sparsity_params") or {}
        self.support_query_data = kwargs.get("support_query_data", "split")
        self.support_batch_mode = kwargs.get("support_batch_mode", "mixed")
        self.query_batch_size = kwargs.get("query_batch_size", 1)
        self.split_query_size = kwargs.get("split_query_size", 0)
        self.split_query_fold = kwargs.get("split_query_fold", 0)
        self.num_iterations = kwargs.get("num_iterations", 1.0)

        self.sparsity_mode_default: list[SparsityMode] = [
            "point",
            "grid",
            "contour",
            "skeleton",
            "region",
        ]
        self.sparsity_mode_additional = self.set_additional_sparse_mode()

        (
            self.num_iterations_int,
            self.support_batches,
            self.support_sparsities,
            self.support_indices,
            self.query_indices,
        ) = self.compose_support_query()

    @abstractmethod
    def set_additional_sparse_mode(self) -> list[SparsityMode]:
        pass

    @abstractmethod
    def get_additional_sparse_mask(
        self,
        sparsity_mode: SparsityMode,
        msk: NDArray,
        img: NDArray | None = None,
        sparsity_value: SparsityValue = "random",
        seed=0,
    ) -> NDArray:
        pass

    def refresh(self, reseed: bool = False) -> None:
        self.cached_items_data = []

        if reseed:
            self.seed = self.seed + int(1e4)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        (
            self.num_iterations_int,
            self.support_batches,
            self.support_sparsities,
            self.support_indices,
            self.query_indices,
        ) = self.compose_support_query()

    @staticmethod
    def sparse_point(
        msk: NDArray,
        num_classes: int,
        sparsity: SparsityValue = "random",
        dot_size: int | None = None,
        seed=0,
    ) -> NDArray:
        np_nrg = np.random.default_rng(seed)

        auto_dot_size = max(min(msk.shape) // 50, 1)
        dot_size = dot_size or auto_dot_size

        sparsity_num = (
            np_nrg.integers(5, 50) if sparsity == "random" else round(sparsity)
        )

        small_msk = FewSparseDataset.resize_image(
            msk, np.divide(msk.shape, dot_size).tolist(), True
        )

        msk_ravel = small_msk.ravel()

        small_msk_point = np.zeros(msk_ravel.shape[0], dtype=msk.dtype)
        small_msk_point[:] = -1

        total_count = msk_ravel.shape[0]
        class_counts = np.unique(msk_ravel, return_counts=True)[1]
        class_ratios = np.sqrt(class_counts / total_count)
        class_points = class_ratios / class_ratios.sum() * sparsity_num
        class_points = np.round(class_points).astype(int)

        for c in range(num_classes):
            msk_class = small_msk_point[msk_ravel == c]
            perm = np_nrg.permutation(msk_class.shape[0])
            msk_class[perm[: min(class_points[c], len(perm))]] = c
            small_msk_point[msk_ravel == c] = msk_class

        small_msk_point = small_msk_point.reshape(small_msk.shape)

        msk_point = FewSparseDataset.resize_image(small_msk_point, msk.shape, True)

        new_msk = np.zeros_like(msk) - 1
        disk_size = dot_size // 3 or 1

        for c in range(num_classes):
            mask_point_c = morphology.binary_erosion(
                msk_point == c, footprint=morphology.disk(disk_size)
            )
            mask_point_c = morphology.binary_dilation(
                mask_point_c, footprint=morphology.disk(disk_size)
            )
            mask_point_c = mask_point_c.astype(np.int8)
            mask_point_c[mask_point_c == 1] = c + 1
            new_msk += mask_point_c

        return new_msk

    @staticmethod
    def sparse_grid(
        msk: NDArray,
        sparsity: SparsityValue = "random",
        spacing: int | None = None,
        dot_size: int | None = None,
        seed=0,
    ) -> NDArray:
        np_nrg = np.random.default_rng(seed)
        prime_num = 11
        blob_seed = seed * prime_num

        auto_spacing = np.max(msk.shape) // 20
        int_spacing = spacing or auto_spacing

        auto_dot_size = max(min(msk.shape) // 80, int_spacing // 5, 1)
        dot_size = dot_size or int(auto_dot_size)

        sparsity_num = np_nrg.uniform() if sparsity == "random" else float(sparsity)

        small_msk = FewSparseDataset.resize_image(
            msk, np.divide(msk.shape, dot_size).tolist(), True
        )
        small_spacing = int_spacing // dot_size

        small_new_msk = np.zeros_like(small_msk)
        small_new_msk[:, :] = -1

        starting = (np_nrg.integers(small_spacing), np_nrg.integers(small_spacing))

        small_new_msk[starting[0] :: small_spacing, starting[1] :: small_spacing] = (
            small_msk[starting[0] :: small_spacing, starting[1] :: small_spacing]
        )

        blobs = skdata.binary_blobs(
            np.max(small_new_msk.shape),
            blob_size_fraction=0.1,
            volume_fraction=sparsity_num,
            rng=blob_seed,
        )
        blobs = blobs[: small_new_msk.shape[0], : small_new_msk.shape[1]]

        small_final_msk = np.zeros_like(small_new_msk)
        small_final_msk[:] = -1
        small_final_msk[blobs] = small_new_msk[blobs]

        final_msk = FewSparseDataset.resize_image(small_final_msk, msk.shape, True)

        return final_msk

    @staticmethod
    def sparse_contour(
        msk: NDArray,
        num_classes: int,
        sparsity: SparsityValue = "random",
        radius_dist: int | None = None,
        radius_thick: int | None = None,
        seed=0,
    ) -> NDArray:
        np_rng = np.random.default_rng(seed)

        sparsity_num = np_rng.uniform() if sparsity == "random" else float(sparsity)

        new_msk = np.zeros_like(msk)

        auto_radius_dist = min(msk.shape) // 60
        radius_dist = radius_dist or auto_radius_dist

        auto_radius_thick = min(msk.shape) // 100
        radius_thick = radius_thick or auto_radius_thick

        selem_dist = morphology.disk(radius_dist)
        selem_thick = morphology.disk(radius_thick)

        for c in range(num_classes):
            msk_class = morphology.binary_erosion(msk == c, selem_dist)
            msk_contr = measure.find_contours(msk_class, 0.0)

            msk_bound = np.zeros_like(msk)

            for _, contour in enumerate(msk_contr):
                rand_rot = np_rng.integers(low=1, high=len(contour))
                for j, coord in enumerate(np.roll(contour, rand_rot, axis=0)):
                    if j < max(
                        1, min(round(len(contour) * sparsity_num), len(contour))
                    ):
                        msk_bound[int(coord[0]), int(coord[1])] = c + 1

            msk_bound = morphology.dilation(msk_bound, footprint=selem_thick)

            msk_bound = msk_bound * (msk == c)

            new_msk += msk_bound

        return new_msk - 1

    @staticmethod
    def sparse_skeleton(
        msk: NDArray,
        num_classes: int,
        sparsity: SparsityValue = "random",
        radius_thick: int | None = None,
        seed=0,
    ) -> NDArray:
        np_rng = np.random.default_rng(seed)
        prime_num = 13
        blob_seed = seed * prime_num

        sparsity_num = np_rng.uniform() if sparsity == "random" else float(sparsity)

        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        auto_radius_thick = min(msk.shape) // 100
        radius_thick = radius_thick or auto_radius_thick
        selem_thick = morphology.disk(radius_thick)

        for c in range(num_classes):
            c_msk = msk == c
            c_skel = morphology.skeletonize(c_msk)
            c_msk = morphology.binary_dilation(c_skel, footprint=selem_thick)

            new_msk[c_msk] = c

        blobs = skdata.binary_blobs(
            np.max(new_msk.shape),
            blob_size_fraction=0.1,
            volume_fraction=sparsity_num,
            rng=blob_seed,
        )
        blobs = blobs[: new_msk.shape[0], : new_msk.shape[1]]

        final_msk = np.zeros_like(new_msk)
        final_msk[:] = -1
        final_msk[blobs] = new_msk[blobs]

        return final_msk

    @staticmethod
    def sparse_region(
        msk: NDArray,
        img: NDArray,
        num_classes: int,
        segments: int | None = 250,
        compactness: float | None = 0.5,
        sparsity: SparsityValue = "random",
        seed=0,
    ) -> NDArray:
        np_rng = np.random.default_rng(seed)

        sparsity_num = np_rng.uniform() if sparsity == "random" else float(sparsity)

        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        segments = segments or 250
        compactness = compactness or 0.5

        slic = segmentation.slic(
            img, n_segments=segments, compactness=compactness, start_label=1
        )
        labels = np.unique(slic)

        pure_regions = [[] for _ in range(num_classes)]
        for label in labels:
            sp = msk[slic == label].ravel()
            cnt = np.bincount(sp)

            for c in range(num_classes):
                if (cnt[c] if c < len(cnt) else None) == cnt.sum():
                    pure_regions[c].append(label)

        for c, pure_region in enumerate(pure_regions):
            perm = np_rng.permutation(len(pure_region))

            perm_last_idx = max(1, round(sparsity_num * len(perm)))
            for sp in np.array(pure_region)[perm[:perm_last_idx]]:
                new_msk[slic == sp] = c

        return new_msk

    def compose_support_query(
        self,
    ) -> tuple[int, list[int], list[SparsityTuple], list[int], list[int]]:
        if (
            self.support_query_data == "split"
            and self.support_batch_mode == "full_permutation"
        ):
            raise ValueError(
                "support_query_data='split' and support_batch_mode='full_permutation' are incompatible"
            )

        if self.support_batch_mode in ["permutation", "full_permutation"]:
            (
                num_iterations_int,
                support_batches,
                support_sparsities,
            ) = self.permute_shot_sparsity_for_support()
        else:
            if isinstance(self.num_iterations, float):
                num_iterations_int = (
                    round(self.num_iterations * len(self.items) * self.split_query_size)
                    // self.query_batch_size
                )
            else:
                num_iterations_int = self.num_iterations
            support_batches = self.make_support_batches(num_iterations_int)
            support_sparsities = []

        support_indices, query_indices = self.make_support_query_indices(
            num_iterations_int, support_batches
        )

        return (
            num_iterations_int,
            support_batches,
            support_sparsities,
            support_indices,
            query_indices,
        )

    def permute_shot_sparsity_for_support(
        self,
    ) -> tuple[int, list[int], list[SparsityTuple]]:
        value_error = "shot_options and values in sparsity_options must be list"
        if not isinstance(self.shot_options, list):
            raise ValueError(value_error)
        sparsity_list_init: list[SparsityTuple] = []
        for sparsity in self.sparsity_options:
            if not isinstance(sparsity[1], list):
                raise ValueError(value_error)
            sparsity_list_init.extend([(sparsity[0], value) for value in sparsity[1]])

        if self.support_batch_mode == "full_permutation":
            num_query_batch = len(self.items) // self.query_batch_size
        else:
            num_query_batch = 1

        num_iterations = len(self.shot_options) * len(sparsity_list_init)
        batch_list: list[int] = []
        sparsity_list: list[SparsityTuple] = []
        for i in range(num_iterations):
            shot = self.shot_options[i // len(sparsity_list_init)]
            sparsity = sparsity_list_init[i % len(sparsity_list_init)]
            batch_list.extend([shot] * num_query_batch)
            sparsity_list.extend([sparsity] * num_query_batch)
        return num_iterations * num_query_batch, batch_list, sparsity_list

    def make_support_batches(self, num_iterations: int) -> list[int]:
        if isinstance(self.shot_options, list) and len(self.shot_options) == 0:
            raise ValueError("shot_options list is empty")

        support_size_init = round((1 - self.split_query_size) * len(self.items))
        if self.shot_options == "all":
            return [support_size_init] * num_iterations

        batch_list = []
        for i in range(num_iterations):
            if self.shot_options == "random":
                shot = self.np_rng.integers(1, support_size_init)
            elif isinstance(self.shot_options, list):
                shot = self.shot_options[i % len(self.shot_options)]
            elif isinstance(self.shot_options, tuple):
                shot = self.np_rng.integers(self.shot_options[0], self.shot_options[1])
            else:
                shot = self.shot_options
            batch_list.append(shot)
        return batch_list

    def make_support_query_indices(
        self, num_iterations: int, support_batches: list[int]
    ) -> tuple[list[int], list[int]]:
        if self.support_query_data == "mixed":
            return self.make_mixed_support_query_indices(
                num_iterations, support_batches
            )
        elif self.support_query_data == "mixed_replaced":
            return self.make_mixed_replaced_support_query_indices(
                num_iterations, support_batches
            )
        elif self.support_query_data == "split":
            return self.make_split_support_query_indices(
                num_iterations, support_batches
            )
        else:
            return [], []

    def make_mixed_support_query_indices(
        self, num_iterations: int, support_batches: list[int]
    ) -> tuple[list[int], list[int]]:
        indices_init = list(range(len(self.items)))
        query_indices = self.extend_data(
            indices_init,
            num_iterations * self.query_batch_size,
            random_state=self.seed,
        )
        support_indices = []
        support_indices_pool = indices_init.copy()
        for i, support_batch in enumerate(support_batches):
            query_indices_batch = query_indices[
                i * self.query_batch_size : (i + 1) * self.query_batch_size
            ]
            remainder = support_batch
            while remainder > 0:
                filtered_pool = list(
                    filter(lambda x: x not in query_indices_batch, support_indices_pool)
                )
                if len(filtered_pool) < remainder:
                    new_indices = filtered_pool
                    remainder -= len(filtered_pool)
                    support_indices_pool.extend(indices_init)
                else:
                    new_indices = self.rng.sample(filtered_pool, remainder)
                    remainder = 0
                for x in new_indices:
                    support_indices_pool.remove(x)
                support_indices.extend(new_indices)
        return support_indices, query_indices

    def make_mixed_replaced_support_query_indices(
        self, num_iterations: int, support_batches: list[int]
    ) -> tuple[list[int], list[int]]:
        indices_init = list(range(len(self.items)))
        query_indices = self.extend_data(
            indices_init,
            num_iterations * self.query_batch_size,
            random_state=self.seed,
        )
        indices_init_set = set(indices_init)
        support_indices = []
        for i, support_batch in enumerate(support_batches):
            query_indices_batch = query_indices[
                i * self.query_batch_size : (i + 1) * self.query_batch_size
            ]
            support_indices_batch = self.rng.sample(
                indices_init_set - set(query_indices_batch), support_batch
            )
            support_indices.extend(support_indices_batch)
        return support_indices, query_indices

    def make_split_support_query_indices(
        self, num_iterations: int, support_batches: list[int]
    ) -> tuple[list[int], list[int]]:
        indices_init = list(range(len(self.items)))
        query_size = floor(self.split_query_size * len(self.items))
        support_indices_init, query_indices_init = self.split_train_test(
            indices_init,
            query_size,
            shuffle=False,
            random_state=self.seed,
            fold=self.split_query_fold,
        )
        support_indices = self.extend_data(
            support_indices_init,
            sum(support_batches),
            random_state=self.seed,
        )
        query_indices = self.extend_data(
            query_indices_init,
            num_iterations * self.query_batch_size,
            random_state=self.seed,
        )
        return support_indices, query_indices

    def select_sparsity(self, index: int) -> tuple[SparsityMode, SparsityValue]:
        rng = random.Random(index)
        sparsity = self.sparsity_options[index % len(self.sparsity_options)]
        sparsity_mode = sparsity[0]
        sparsity_value_options = sparsity[1]
        if isinstance(sparsity_value_options, list):
            sparsity_value = rng.choice(sparsity_value_options)
        elif isinstance(sparsity_value_options, tuple):
            low, high = sparsity_value_options
            if isinstance(low, float) or isinstance(high, float):
                sparsity_value = rng.uniform(low, high)
            else:
                sparsity_value = rng.randint(low, high)
        elif sparsity_value_options == "random":
            sparsity_value = "random"
        else:
            sparsity_value = sparsity_value_options
        return sparsity_mode, sparsity_value

    def get_sparse_mask(
        self,
        sparsity_mode: SparsityMode,
        msk: NDArray,
        img: NDArray | None = None,
        sparsity_value: SparsityValue = "random",
        seed=0,
    ) -> NDArray:
        sparse_msk = np.copy(msk)

        if sparsity_mode == "random":
            selected_sparsity_mode = self.rng.choice(
                self.sparsity_mode_default + self.sparsity_mode_additional
            )
            selected_sparsity_value = "random"
        else:
            selected_sparsity_mode = sparsity_mode
            selected_sparsity_value = sparsity_value

        if selected_sparsity_mode in self.sparsity_mode_additional:
            sparse_msk = self.get_additional_sparse_mask(
                selected_sparsity_mode, msk, img, selected_sparsity_value, seed
            )

        elif selected_sparsity_mode == "point":
            sparse_msk = self.sparse_point(
                msk,
                self.num_classes,
                sparsity=selected_sparsity_value,
                seed=seed,
                dot_size=self.sparsity_params.get("point_dot_size"),
            )
        elif selected_sparsity_mode == "grid":
            sparse_msk = self.sparse_grid(
                msk,
                sparsity=selected_sparsity_value,
                seed=seed,
                spacing=self.sparsity_params.get("grid_spacing"),
                dot_size=self.sparsity_params.get("grid_dot_size"),
            )
        elif selected_sparsity_mode == "contour":
            sparse_msk = self.sparse_contour(
                msk,
                self.num_classes,
                sparsity=selected_sparsity_value,
                seed=seed,
                radius_dist=self.sparsity_params.get("contour_radius_dist"),
                radius_thick=self.sparsity_params.get("contour_radius_thick"),
            )
        elif selected_sparsity_mode == "skeleton":
            sparse_msk = self.sparse_skeleton(
                msk,
                self.num_classes,
                sparsity=selected_sparsity_value,
                seed=seed,
                radius_thick=self.sparsity_params.get("skeleton_radius_thick"),
            )
        elif selected_sparsity_mode == "region" and img is not None:
            sparse_msk = self.sparse_region(
                msk,
                img,
                self.num_classes,
                sparsity=selected_sparsity_value,
                seed=seed,
                segments=self.sparsity_params.get("region_segments"),
                compactness=self.sparsity_params.get("region_compactness"),
            )

        return sparse_msk

    def get_data_with_sparse_all(
        self,
        index: int,
        sparsity_values: dict[str, SparsityValue] | None = None,
    ) -> tuple[NDArray, NDArray, dict[str, NDArray], str]:
        img, msk, img_filename = self.get_data(index)

        all_sparse_msk = {}

        for sparsity_mode in self.sparsity_mode_default + self.sparsity_mode_additional:
            sparsity_value = (
                sparsity_values.get(sparsity_mode, "random")
                if sparsity_values is not None
                else "random"
            )
            all_sparse_msk[sparsity_mode] = self.get_sparse_mask(
                sparsity_mode, msk, img, sparsity_value, index
            )

        return img, msk, all_sparse_msk, img_filename

    def get_support_data(
        self,
        index: int,
        sparsity_mode: SparsityMode | None = None,
        sparsity_value: SparsityValue | None = None,
    ) -> tuple[NDArray, NDArray, int, SparsityMode, SparsityValue]:
        img_idx = self.support_indices[index]
        img, msk, _ = self.get_data(img_idx)
        if sparsity_mode is None or sparsity_value is None:
            sparsity_mode, sparsity_value = self.select_sparsity(index)

        sparse_msk = self.get_sparse_mask(
            sparsity_mode,
            msk,
            img,
            sparsity_value,
            index,
        )

        return img, sparse_msk, img_idx, sparsity_mode, sparsity_value

    def get_query_data(self, index: int) -> tuple[NDArray, NDArray, int]:
        img_idx = self.query_indices[index]
        img, msk, _ = self.get_data(img_idx)
        return img, msk, img_idx

    def __getitem__(self, index: int) -> FewSparseDataTuple:
        support_batch_size = self.support_batches[index]
        support_images_list = []
        support_masks_list = []
        support_indices = []
        support_sparsity_modes: list[SparsityMode] = []
        support_sparsity_values: list[SparsityValue] = []

        if self.support_batch_mode in ["permutation", "full_permutation"]:
            sparsity_mode, sparsity_value = self.support_sparsities[index]
        elif self.support_batch_mode == "homogen":
            sparsity_mode, sparsity_value = self.select_sparsity(index)
        else:
            sparsity_mode, sparsity_value = None, None

        for i in range(support_batch_size):
            item_index = sum(self.support_batches[:index]) + i
            img, msk, img_idx, mode, value = self.get_support_data(
                item_index, sparsity_mode, sparsity_value
            )
            img = self.prepare_image_as_tensor(img)
            msk = self.prepare_mask_as_tensor(msk)
            support_images_list.append(img)
            support_masks_list.append(msk)
            support_indices.append(img_idx)
            if self.support_batch_mode == "mixed":
                support_sparsity_modes.append(mode)
                support_sparsity_values.append(value)

        if self.support_batch_mode == "mixed":
            sparsity_mode = support_sparsity_modes
            sparsity_value = support_sparsity_values

        assert sparsity_mode is not None and sparsity_value is not None

        query_images_list = []
        query_masks_list = []
        query_indices = []
        for i in range(self.query_batch_size):
            item_index = self.query_batch_size * index + i
            img, msk, img_idx = self.get_query_data(item_index)
            img = self.prepare_image_as_tensor(img)
            msk = self.prepare_mask_as_tensor(msk)
            query_images_list.append(img)
            query_masks_list.append(msk)
            query_indices.append(img_idx)

        support_images = torch.stack(support_images_list, dim=0)
        support_masks = torch.stack(support_masks_list, dim=0)
        query_images = torch.stack(query_images_list, dim=0)
        query_masks = torch.stack(query_masks_list, dim=0)

        return FewSparseDataTuple(
            support=SupportDataTuple(
                images=support_images,
                masks=support_masks,
                indices=support_indices,
                sparsity_mode=sparsity_mode,
                sparsity_value=sparsity_value,
            ),
            query=QueryDataTuple(
                images=query_images, masks=query_masks, indices=query_indices
            ),
            dataset_name=self.dataset_name,
        )

    def __len__(self):
        return len(self.support_batches)
