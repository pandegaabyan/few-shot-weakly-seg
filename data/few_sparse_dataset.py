import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray
from skimage import data as skdata
from skimage import measure
from skimage import morphology
from skimage import segmentation
from skimage import transform
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data.types import SparsityModes, SparsityValue, DatasetModes, TensorDataItem


class FewSparseDataset(Dataset, ABC):

    def __init__(self,
                 mode: DatasetModes,
                 num_classes: int,
                 resize_to: tuple[int, int],
                 num_shots: int = -1,
                 split_seed: int | None = None,
                 split_test_size: float = 0.2,
                 sparsity_mode: SparsityModes = 'dense',
                 sparsity_value: SparsityValue = 'random',
                 sparsity_params: dict | None = None):

        # Initializing variables.
        self.mode = mode
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.resize_to = resize_to
        self.split_seed = split_seed
        self.split_test_size = split_test_size
        self.sparsity_mode = sparsity_mode
        self.sparsity_value = sparsity_value
        self.sparsity_params = sparsity_params or {}

        self.sparsity_mode_default: list[SparsityModes] = ["point", "grid", "contour", "skeleton", "region"]
        self.sparsity_mode_additional: list[SparsityModes] = self.set_additional_sparse_mode()

        # Creating list of paths.
        self.items = self.make_data_list()
        if len(self.items) == 0:
            raise (RuntimeError('Found 0 items, please check the dataset'))

    # Function that create the list of pairs (img_path, mask_path)
    # Implement this function for your dataset structure
    @abstractmethod
    def get_all_data_path(self) -> list[tuple[str, str]]:
        pass

    @staticmethod
    @abstractmethod
    def read_image_mask(img_path: str, msk_path: str) -> tuple[NDArray, NDArray]:
        # image should have shape (H, W, C) and dtype uint8 while mask should have shape (H, W) and dtype int8
        pass

    @abstractmethod
    def set_additional_sparse_mode(self) -> list[SparsityModes]:
        pass

    @abstractmethod
    def get_additional_sparse_mask(self, sparsity_mode: SparsityModes, msk: NDArray, img: NDArray | None = None,
                                   sparsity_value: SparsityValue = 'random', seed=0) -> NDArray:
        pass

    @staticmethod
    def sparse_point(msk: NDArray, num_classes: int, sparsity: SparsityValue = "random",
                     dot_size: int | None = None, seed=0) -> NDArray:
        if sparsity != 'random':
            np.random.seed(seed)

        default_dot_size = max(min(msk.shape) // 50, 1)
        dot_size = dot_size or default_dot_size

        small_msk = FewSparseDataset.resize_image(msk, np.divide(msk.shape, dot_size).tolist(), True)

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
            if type(sparsity) is float or type(sparsity) is int:
                sparsity_num = round(sparsity)
            elif type(sparsity) is tuple:
                sparsity_num = round(np.random.uniform(low=sparsity[0], high=sparsity[1]))
            else:
                sparsity_num = np.random.randint(low=1, high=len(perm))
            msk_class[perm[:min(sparsity_num, len(perm))]] = c

            # Merging sparse masks.
            small_msk_point[msk_ravel == c] = msk_class

        # Reshaping linearized sparse mask to the original 2 dimensions.
        small_msk_point = small_msk_point.reshape(small_msk.shape)

        msk_point = FewSparseDataset.resize_image(small_msk_point, msk.shape, True)

        new_msk = np.zeros_like(msk) - 1
        disk_size = dot_size // 3 or 1

        for c in range(num_classes):
            mask_point_c = morphology.binary_erosion(msk_point == c, footprint=morphology.disk(disk_size))
            mask_point_c = morphology.binary_dilation(mask_point_c, footprint=morphology.disk(disk_size))
            mask_point_c = mask_point_c.astype(np.int8)
            mask_point_c[mask_point_c == 1] = c + 1
            new_msk += mask_point_c

        np.random.seed(None)

        return new_msk

    @staticmethod
    def sparse_grid(msk: NDArray, sparsity: SparsityValue = "random", dot_size: int | None = None, seed=0) -> NDArray:
        if sparsity != 'random':
            np.random.seed(seed)

        default_dot_size = max(
            min(msk.shape) // 80,
            sparsity // 5 if (type(sparsity) is float or type(sparsity) is int) else 0,
            1
        )
        dot_size = dot_size or default_dot_size

        small_msk = FewSparseDataset.resize_image(msk, np.divide(msk.shape, dot_size).tolist(), True)

        # Copying mask and starting it with -1 for inserting sparsity.
        small_new_msk = np.zeros_like(small_msk)
        small_new_msk[:, :] = -1

        if type(sparsity) is float or type(sparsity) is int:
            # Predetermined sparsity (x and y point spacing).
            spacing_value = int(sparsity / dot_size)
        elif type(sparsity) is tuple:
            spacing_value = round(np.random.uniform(low=sparsity[0], high=sparsity[1]))
        else:
            # Random sparsity (x and y point spacing).
            max_high = int(np.max(small_msk.shape) / 2)
            spacing_value = np.random.randint(low=1, high=max_high)
        spacing = (spacing_value, spacing_value)

        starting = (np.random.randint(spacing[0]),
                    np.random.randint(spacing[1]))

        small_new_msk[starting[0]::spacing[0], starting[1]::spacing[1]] = \
            small_msk[starting[0]::spacing[0], starting[1]::spacing[1]]

        new_msk = FewSparseDataset.resize_image(small_new_msk, msk.shape, True)

        np.random.seed(None)

        return new_msk

    @staticmethod
    def sparse_contour(msk: NDArray, num_classes: int, sparsity: SparsityValue = 'random',
                       radius_dist: int | None = None, radius_thick: int | None = None, seed=0) -> NDArray:
        if sparsity != 'random':
            np.random.seed(seed)

        if type(sparsity) is float or type(sparsity) is int:
            sparsity_num = sparsity
        elif type(sparsity) is tuple:
            sparsity_num = np.random.uniform(low=sparsity[0], high=sparsity[1])
        else:
            sparsity_num = np.random.uniform()

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
                    if j < max(1, min(round(len(contour) * sparsity_num), len(contour))):
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
    def sparse_skeleton(msk: NDArray, num_classes: int, sparsity: SparsityValue = 'random',
                        radius_thick: int | None = None, seed=0) -> NDArray:
        bseed = None  # Blobs generator seed
        if sparsity != 'random':
            np.random.seed(seed)
            bseed = seed

        if type(sparsity) is float or type(sparsity) is int:
            sparsity_num = sparsity
        elif type(sparsity) is tuple:
            sparsity_num = np.random.uniform(low=sparsity[0], high=sparsity[1])
        else:
            sparsity_num = np.random.uniform()

        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Randomly selecting disk radius the annotation thickness.
        radius_thick = radius_thick or 1
        selem_thick = morphology.disk(radius_thick)

        for c in range(num_classes):
            c_msk = (msk == c)
            c_skel = morphology.skeletonize(c_msk)
            c_msk = morphology.binary_dilation(c_skel, footprint=selem_thick)

            new_msk[c_msk] = c

        blobs = skdata.binary_blobs(np.max(new_msk.shape), blob_size_fraction=0.1,
                                    volume_fraction=sparsity_num, seed=bseed)
        blobs = blobs[:new_msk.shape[0], :new_msk.shape[1]]

        n_sp = np.zeros_like(new_msk)
        n_sp[:] = -1
        n_sp[blobs] = new_msk[blobs]

        np.random.seed(None)

        return n_sp

    @staticmethod
    def sparse_region(msk: NDArray, img: NDArray, num_classes: int, compactness: float | None = 0.5,
                      sparsity: SparsityValue = 'random', seed=0) -> NDArray:
        if sparsity != "random":
            np.random.seed(seed)

        if type(sparsity) is float or type(sparsity) is int:
            sparsity_num = sparsity
        elif type(sparsity) is tuple:
            sparsity_num = np.random.uniform(low=sparsity[0], high=sparsity[1])
        else:
            sparsity_num = np.random.uniform()

        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Computing SLIC super pixels.
        slic = segmentation.slic(
            img, n_segments=250, compactness=compactness or 0.5, start_label=1)
        labels = np.unique(slic)

        # Finding 'pure' regions, that is, the ones that only contain one label within.
        pure_regions = [[] for _ in range(num_classes)]
        for label in labels:
            sp = msk[slic == label].ravel()
            cnt = np.bincount(sp)

            for c in range(num_classes):
                if (cnt[c] if c < len(cnt) else None) == cnt.sum():
                    pure_regions[c].append(label)

        for (c, pure_region) in enumerate(pure_regions):
            # Random permutation to pure region.
            perm = np.random.permutation(len(pure_region))

            # Only keeping the selected k regions.
            perm_last_idx = max(1, round(sparsity_num * len(perm)))
            for sp in np.array(pure_region)[perm[:perm_last_idx]]:
                new_msk[slic == sp] = c

        np.random.seed(None)

        return new_msk

    @staticmethod
    def norm(img: NDArray) -> NDArray:
        normalized = np.zeros(img.shape)
        if len(img.shape) == 2:
            normalized = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                normalized[:, :, b] = (img[:, :, b] - img[:, :, b].mean()) / img[:, :, b].std()
        return normalized.astype(np.float32)

    @staticmethod
    def ensure_channels(img: NDArray) -> NDArray:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    @staticmethod
    def resize_image(img: NDArray, resize_to: tuple[int, ...], is_mask: bool):
        if is_mask:
            order = 0
            anti_aliasing = False
        else:
            order = 1
            anti_aliasing = True

        resized = transform.resize(img, resize_to, order=order,
                                   preserve_range=True, anti_aliasing=anti_aliasing)
        resized = resized.astype(img.dtype)

        return resized

    @staticmethod
    def filename_from_path(path: str) -> str:
        # Splitting path.
        filename = os.path.split(path)[-1]

        # remove extension from filename
        filename = ".".join(filename.split(".")[:-1])

        return filename

    def make_data_list(self) -> list[tuple[str, str]]:
        # Split the data in train/val
        tr, ts = train_test_split(self.get_all_data_path(), test_size=self.split_test_size,
                                  random_state=self.split_seed, shuffle=False)

        # Select split, based on the mode
        if 'train' in self.mode:
            data_list = tr
        elif 'test' in self.mode:
            data_list = ts
        else:
            return []

        random.seed(self.split_seed)
        random.shuffle(data_list)
        random.seed(None)

        # If few-shot, select only a subset of samples
        if self.num_shots != -1 and self.num_shots <= len(data_list):
            data_list = data_list[:self.num_shots]

        # Returning list.
        return data_list

    def get_sparse_mask(self, sparsity_mode: SparsityModes, msk: NDArray, img: NDArray | None = None,
                        sparsity_value: SparsityValue = 'random', seed=0) -> NDArray:

        sparse_msk = np.copy(msk)

        if sparsity_mode == 'random':
            selected_sparsity_mode = random.choice(self.sparsity_mode_default)
            selected_sparsity_value = "random"
        else:
            selected_sparsity_mode = sparsity_mode
            selected_sparsity_value = sparsity_value

        if selected_sparsity_mode in self.sparsity_mode_additional:
            sparse_msk = self.get_additional_sparse_mask(
                selected_sparsity_mode, msk, img, selected_sparsity_value, seed)

        elif selected_sparsity_mode == 'point':
            sparse_msk = self.sparse_point(
                msk, self.num_classes, sparsity=selected_sparsity_value, seed=seed,
                dot_size=self.sparsity_params.get('point_dot_size'))
        elif selected_sparsity_mode == 'grid':
            sparse_msk = self.sparse_grid(
                msk, sparsity=selected_sparsity_value, seed=seed,
                dot_size=self.sparsity_params.get('grid_dot_size'))
        elif selected_sparsity_mode == 'contour':
            sparse_msk = self.sparse_contour(
                msk, self.num_classes, sparsity=selected_sparsity_value, seed=seed,
                radius_dist=self.sparsity_params.get('contour_radius_dist'),
                radius_thick=self.sparsity_params.get('contour_radius_thick'))
        elif selected_sparsity_mode == 'skeleton':
            sparse_msk = self.sparse_skeleton(
                msk, self.num_classes, sparsity=selected_sparsity_value, seed=seed,
                radius_thick=self.sparsity_params.get('skeleton_radius_thick'))
        elif selected_sparsity_mode == 'region' and img is not None:
            sparse_msk = self.sparse_region(
                msk, img, self.num_classes, sparsity=selected_sparsity_value, seed=seed,
                compactness=self.sparsity_params.get('region_compactness'))

        return sparse_msk

    # Function to load images and masks
    # Implement this function based on your data
    # Returns: img, mask, img_filename
    def get_data(self, index: int) -> tuple[NDArray, NDArray, str]:
        img_path, msk_path = self.items[index]

        img, msk = self.read_image_mask(img_path, msk_path)

        img = self.resize_image(img, self.resize_to, False)
        msk = self.resize_image(msk, self.resize_to, True)

        img_filename = self.filename_from_path(img_path)

        return img, msk, img_filename

    def get_data_with_sparse(self, index: int) -> tuple[NDArray, NDArray, NDArray, str]:
        img, msk, img_filename = self.get_data(index)

        sparse_msk = self.get_sparse_mask(self.sparsity_mode, msk, img,
                                          self.sparsity_value, index)  # type: ignore

        # Returning to iterator.
        return img, msk, sparse_msk, img_filename

    def get_data_with_sparse_all(self,
                                 index: int,
                                 sparsity_values: dict[str, SparsityValue] | None = None,
                                 ) -> tuple[NDArray, NDArray, dict[str, NDArray], str]:
        img, msk, img_filename = self.get_data(index)

        all_sparse_msk = {}

        for sparsity_mode in self.sparsity_mode_default + self.sparsity_mode_additional:
            sparsity_value = sparsity_values.get(sparsity_mode, "random") if sparsity_values is not None else "random"
            all_sparse_msk[sparsity_mode] = self.get_sparse_mask(sparsity_mode, msk, img,
                                                                 sparsity_value, index)

        return img, msk, all_sparse_msk, img_filename

    def __getitem__(self, index: int) -> TensorDataItem:

        img, msk, sparse_msk, img_filename = self.get_data_with_sparse(index)

        # Normalization.
        img = self.norm(img)

        # Ensure image has channel dimension.
        img = self.ensure_channels(img)

        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk).type(torch.int64)

        sparse_msk = torch.from_numpy(sparse_msk).type(torch.int64)

        # Returning to iterator.
        return img, msk, sparse_msk, img_filename

    def __len__(self):
        return len(self.items)
