import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import data as skdata
from skimage import measure, morphology, segmentation

from data.base_dataset import BaseDataset
from data.typings import (
    DatasetModes,
    FewSparseDataTuple,
    QueryDataTuple,
    ShotOptions,
    SparsityMode,
    SparsityOptions,
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
        max_items: int | None = None,
        seed: int | None = None,
        split_val_size: float = 0,
        split_test_size: float = 0,
        cache_data: bool = False,
        dataset_name: str | None = None,
        shot_options: ShotOptions = "all",
        sparsity_options: SparsityOptions = [("random", "random")],
        sparsity_params: dict | None = None,
        shot_sparsity_permutation: bool = False,
        homogen_support_batch: bool = False,
        query_batch_size: int = 1,
        split_query_size: float = 0,
        split_query_fold: int = 0,
        num_iterations: int | float = 1.0,
    ):
        super().__init__(
            mode,
            num_classes,
            resize_to,
            max_items,
            seed,
            split_val_size,
            0,
            split_test_size,
            0,
            cache_data,
            dataset_name,
        )

        # Initializing variables.
        self.shot_options: ShotOptions = shot_options
        self.sparsity_options: SparsityOptions = sparsity_options
        self.sparsity_params = sparsity_params or {}
        self.shot_sparsity_permutation = shot_sparsity_permutation
        self.homogen_support_batch = homogen_support_batch or shot_sparsity_permutation
        self.query_batch_size = query_batch_size
        self.split_query_size = split_query_size
        self.split_query_fold = split_query_fold

        if self.shot_sparsity_permutation:
            (
                self.num_iterations,
                self.support_batches,
                self.support_sparsities,
            ) = self.permute_shot_sparsity_for_support()
        else:
            if isinstance(num_iterations, float):
                num_iterations = (
                    round(num_iterations * len(self.items) * split_query_size)
                    // query_batch_size
                )
            self.num_iterations = num_iterations
            self.support_batches = self.make_support_batches()
            self.support_sparsities = []

        self.support_indices, self.query_indices = self.make_support_query_indices()

        self.sparsity_mode_default: list[SparsityMode] = [
            "point",
            "grid",
            "contour",
            "skeleton",
            "region",
        ]
        self.sparsity_mode_additional = self.set_additional_sparse_mode()

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

    @staticmethod
    def sparse_point(
        msk: NDArray,
        num_classes: int,
        sparsity: SparsityValue = "random",
        dot_size: int | None = None,
        seed=0,
    ) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        default_dot_size = max(min(msk.shape) // 50, 1)
        dot_size = dot_size or default_dot_size

        small_msk = FewSparseDataset.resize_image(
            msk, np.divide(msk.shape, dot_size).tolist(), True
        )

        # Linearizing mask.
        msk_ravel = small_msk.ravel()

        # Copying raveled mask and starting it with -1 for inserting sparsity.
        small_msk_point = np.zeros(msk_ravel.shape[0], dtype=msk.dtype)
        small_msk_point[:] = -1

        for c in range(num_classes):
            # Slicing array for only containing class "c" pixels.
            msk_class = small_msk_point[msk_ravel == c]

            # Random permutation of class "c" pixels.
            perm = np.random.permutation(msk_class.shape[0])
            if sparsity == "random":
                sparsity_num = np.random.randint(low=1, high=len(perm))
            else:
                sparsity_num = round(sparsity)
            msk_class[perm[: min(sparsity_num, len(perm))]] = c

            # Merging sparse masks.
            small_msk_point[msk_ravel == c] = msk_class

        # Reshaping linearized sparse mask to the original 2 dimensions.
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

        np.random.seed(None)

        return new_msk

    @staticmethod
    def sparse_grid(
        msk: NDArray,
        sparsity: SparsityValue = "random",
        dot_size: int | None = None,
        seed=0,
    ) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        default_dot_size = max(
            min(msk.shape) // 80,
            # sparsity // 5 if (isinstance(sparsity, float) or isinstance(sparsity, int)) else 0,
            sparsity // 5
            if (isinstance(sparsity, float) or isinstance(sparsity, int))
            else 0,
            1,
        )
        dot_size = dot_size or int(default_dot_size)

        small_msk = FewSparseDataset.resize_image(
            msk, np.divide(msk.shape, dot_size).tolist(), True
        )

        # Copying mask and starting it with -1 for inserting sparsity.
        small_new_msk = np.zeros_like(small_msk)
        small_new_msk[:, :] = -1

        if sparsity == "random":
            # Random sparsity (x and y point spacing).
            max_high = int(np.max(small_msk.shape) / 2)
            spacing_value = np.random.randint(low=1, high=max_high)
        else:
            # Predetermined sparsity (x and y point spacing).
            spacing_value = int(sparsity / dot_size)
        spacing = (spacing_value, spacing_value)

        starting = (np.random.randint(spacing[0]), np.random.randint(spacing[1]))

        small_new_msk[starting[0] :: spacing[0], starting[1] :: spacing[1]] = small_msk[
            starting[0] :: spacing[0], starting[1] :: spacing[1]
        ]

        new_msk = FewSparseDataset.resize_image(small_new_msk, msk.shape, True)

        np.random.seed(None)

        return new_msk

    @staticmethod
    def sparse_contour(
        msk: NDArray,
        num_classes: int,
        sparsity: SparsityValue = "random",
        radius_dist: int | None = None,
        radius_thick: int | None = None,
        seed=0,
    ) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        sparsity_num = np.random.uniform() if sparsity == "random" else float(sparsity)

        new_msk = np.zeros_like(msk)

        # Random disk radius for erosions and dilations from the original mask.
        radius_dist = radius_dist or min(msk.shape) // 60

        # Random disk radius for annotation thickness.
        radius_thick = radius_thick or 1

        # Creating morphology elements.
        selem_dist = morphology.disk(radius_dist)
        selem_thick = morphology.disk(radius_thick)

        for c in range(num_classes):
            # Eroding original mask and obtaining contour.
            msk_class = morphology.binary_erosion(msk == c, selem_dist)
            msk_contr = measure.find_contours(msk_class, 0.0)

            # Instantiating masks for the boundaries.
            msk_bound = np.zeros_like(msk)

            # Filling boundary masks.
            for _, contour in enumerate(msk_contr):
                rand_rot = np.random.randint(low=1, high=len(contour))
                for j, coord in enumerate(np.roll(contour, rand_rot, axis=0)):
                    if j < max(
                        1, min(round(len(contour) * sparsity_num), len(contour))
                    ):
                        msk_bound[int(coord[0]), int(coord[1])] = c + 1

            # Dilating boundary masks to make them thicker.
            msk_bound = morphology.dilation(msk_bound, footprint=selem_thick)

            # Removing invalid boundary masks.
            msk_bound = msk_bound * (msk == c)

            # Merging boundary masks.
            new_msk += msk_bound

        np.random.seed(None)

        return new_msk - 1

    @staticmethod
    def sparse_skeleton(
        msk: NDArray,
        num_classes: int,
        sparsity: SparsityValue = "random",
        radius_thick: int | None = None,
        seed=0,
    ) -> NDArray:
        bseed = None  # Blobs generator seed
        if sparsity != "random":
            np.random.seed(seed)
            bseed = seed

        sparsity_num = np.random.uniform() if sparsity == "random" else float(sparsity)

        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Randomly selecting disk radius the annotation thickness.
        radius_thick = radius_thick or 1
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
            seed=bseed,
        )
        blobs = blobs[: new_msk.shape[0], : new_msk.shape[1]]

        n_sp = np.zeros_like(new_msk)
        n_sp[:] = -1
        n_sp[blobs] = new_msk[blobs]

        np.random.seed(None)

        return n_sp

    @staticmethod
    def sparse_region(
        msk: NDArray,
        img: NDArray,
        num_classes: int,
        compactness: float | None = 0.5,
        sparsity: SparsityValue = "random",
        seed=0,
    ) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        sparsity_num = np.random.uniform() if sparsity == "random" else float(sparsity)

        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Computing SLIC super pixels.
        slic = segmentation.slic(
            img, n_segments=250, compactness=compactness or 0.5, start_label=1
        )
        labels = np.unique(slic)

        # Finding 'pure' regions, that is, the ones that only contain one label within.
        pure_regions = [[] for _ in range(num_classes)]
        for label in labels:
            sp = msk[slic == label].ravel()
            cnt = np.bincount(sp)

            for c in range(num_classes):
                if (cnt[c] if c < len(cnt) else None) == cnt.sum():
                    pure_regions[c].append(label)

        for c, pure_region in enumerate(pure_regions):
            # Random permutation to pure region.
            perm = np.random.permutation(len(pure_region))

            # Only keeping the selected k regions.
            perm_last_idx = max(1, round(sparsity_num * len(perm)))
            for sp in np.array(pure_region)[perm[:perm_last_idx]]:
                new_msk[slic == sp] = c

        np.random.seed(None)

        return new_msk

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
        num_iterations = len(self.shot_options) * len(sparsity_list_init)
        batch_list: list[int] = []
        sparsity_list: list[SparsityTuple] = []
        for i in range(num_iterations):
            shot = self.shot_options[i // len(sparsity_list_init)]
            sparsity = sparsity_list_init[i % len(sparsity_list_init)]
            batch_list.append(shot)
            sparsity_list.append(sparsity)
        return num_iterations, batch_list, sparsity_list

    def make_support_batches(self) -> list[int]:
        if isinstance(self.shot_options, list) and len(self.shot_options) == 0:
            raise ValueError("shot_options list is empty")

        support_size_init = round((1 - self.split_query_size) * len(self.items))
        if self.shot_options == "all":
            return [support_size_init] * self.num_iterations

        batch_list = []
        for i in range(self.num_iterations):
            if self.shot_options == "random":
                shot = np.random.randint(1, support_size_init)
            elif isinstance(self.shot_options, list):
                shot = self.shot_options[i % len(self.shot_options)]
            elif isinstance(self.shot_options, tuple):
                np.random.seed(i)
                shot = np.random.randint(self.shot_options[0], self.shot_options[1])
                np.random.seed(None)
            else:
                shot = self.shot_options
            batch_list.append(shot)
        return batch_list

    def make_support_query_indices(self) -> tuple[list[int], list[int]]:
        query_size = round(self.split_query_size * len(self.items))
        support_indices_init, query_indices_init = self.split_train_test(
            list(range(len(self.items))),
            test_size=query_size,
            random_state=self.seed,
            shuffle=False,
            fold=self.split_query_fold,
        )
        support_indices = self.extend_data(
            support_indices_init, sum(self.support_batches)
        )
        query_indices = self.extend_data(
            query_indices_init, self.num_iterations * self.query_batch_size
        )
        return support_indices, query_indices

    def select_sparsity(self, index: int) -> tuple[SparsityMode, SparsityValue]:
        if len(self.sparsity_options) == 0:
            return "random", "random"
        random.seed(index)
        sparsity = self.sparsity_options[index % len(self.sparsity_options)]
        sparsity_mode = sparsity[0]
        sparsity_value_options = sparsity[1]
        if isinstance(sparsity_value_options, list):
            sparsity_value = random.choice(sparsity_value_options)
        elif isinstance(sparsity_value_options, tuple):
            low, high = sparsity_value_options
            if isinstance(low, float) or isinstance(high, float):
                sparsity_value = random.uniform(low, high)
            else:
                sparsity_value = random.randint(low, high)
        elif sparsity_value_options == "random":
            sparsity_value = "random"
        else:
            sparsity_value = sparsity_value_options
        random.seed(None)
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
            selected_sparsity_mode = random.choice(
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
    ) -> tuple[NDArray, NDArray, str, SparsityMode, SparsityValue]:
        img, msk, img_filename = self.get_data(self.support_indices[index])
        if sparsity_mode is None or sparsity_value is None:
            sparsity_mode, sparsity_value = self.select_sparsity(index)

        sparse_msk = self.get_sparse_mask(
            sparsity_mode,
            msk,
            img,
            sparsity_value,
            index,
        )

        return img, sparse_msk, img_filename, sparsity_mode, sparsity_value

    def get_query_data(self, index: int) -> tuple[NDArray, NDArray, str]:
        return self.get_data(self.query_indices[index])

    def __getitem__(self, index: int) -> FewSparseDataTuple:
        support_batch_size = self.support_batches[index]
        support_images_list = []
        support_masks_list = []
        support_name_list = []
        support_sparsity_modes: list[SparsityMode] = []
        support_sparsity_values: list[SparsityValue] = []
        if self.homogen_support_batch:
            if self.shot_sparsity_permutation:
                sparsity_mode, sparsity_value = self.support_sparsities[index]
            else:
                sparsity_mode, sparsity_value = self.select_sparsity(index)
        else:
            sparsity_mode, sparsity_value = None, None
        for i in range(support_batch_size):
            item_index = sum(self.support_batches[:index]) + i
            img, msk, name, mode, value = self.get_support_data(
                item_index, sparsity_mode, sparsity_value
            )
            img = self.prepare_image_as_tensor(img)
            msk = self.prepare_mask_as_tensor(msk)
            support_images_list.append(img)
            support_masks_list.append(msk)
            support_name_list.append(name)
            if not self.homogen_support_batch:
                support_sparsity_modes.append(mode)
                support_sparsity_values.append(value)
        if not self.homogen_support_batch:
            sparsity_mode = support_sparsity_modes
            sparsity_value = support_sparsity_values
        assert sparsity_mode is not None and sparsity_value is not None

        query_images_list = []
        query_masks_list = []
        query_names_list = []
        for i in range(self.query_batch_size):
            item_index = self.query_batch_size * index + i
            img, msk, name = self.get_query_data(item_index)
            img = self.prepare_image_as_tensor(img)
            msk = self.prepare_mask_as_tensor(msk)
            query_images_list.append(img)
            query_masks_list.append(msk)
            query_names_list.append(name)

        support_images = torch.stack(support_images_list, dim=0)
        support_masks = torch.stack(support_masks_list, dim=0)
        query_images = torch.stack(query_images_list, dim=0)
        query_masks = torch.stack(query_masks_list, dim=0)

        return FewSparseDataTuple(
            support=SupportDataTuple(
                images=support_images,
                masks=support_masks,
                file_names=support_name_list,
                sparsity_mode=sparsity_mode,
                sparsity_value=sparsity_value,
            ),
            query=QueryDataTuple(
                images=query_images, masks=query_masks, file_names=query_names_list
            ),
            dataset_name=self.dataset_name,
        )

    def __len__(self):
        return len(self.support_batches)
