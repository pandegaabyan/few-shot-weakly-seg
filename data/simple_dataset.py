import random
from abc import ABC

import torch
from sklearn.model_selection import train_test_split

from data.base_dataset import BaseDataset
from data.types import SimpleDatasetModes, SimpleTensorDataItem


class SimpleDataset(BaseDataset, ABC):

    def __init__(self,
                 mode: SimpleDatasetModes,
                 num_classes: int,
                 resize_to: tuple[int, int],
                 num_shots: int = -1,
                 split_seed: int | None = None,
                 split_val_size: float = 0.2,
                 split_test_size: float = 0.2):

        # Initializing variables.
        self.mode = mode
        self.num_shots = num_shots
        self.split_seed = split_seed
        self.split_val_size = split_val_size
        self.split_test_size = split_test_size

        super().__init__(num_classes, resize_to)

    def make_data_list(self) -> list[tuple[str, str]]:
        # Splitting data.
        val_test_size = self.split_val_size + self.split_test_size
        tr, val_ts = self.split_train_test(self.get_all_data_path(), test_size=val_test_size,
                                           random_state=self.split_seed, shuffle=False)
        val, ts = self.split_train_test(val_ts, test_size=self.split_test_size / val_test_size,
                                        random_state=self.split_seed, shuffle=False)

        # Select split, based on the mode
        if 'train' in self.mode:
            data_list = tr
        elif 'val' in self.mode:
            data_list = val
        elif 'test' in self.mode:
            data_list = ts
        else:
            return []

        random.seed(self.split_seed)
        random.shuffle(data_list)
        random.seed(None)

        return data_list

    def __getitem__(self, index: int) -> SimpleTensorDataItem:

        img, msk, img_filename = self.get_data(index)

        # Normalization.
        img = self.norm(img)

        # Ensure image has channel dimension.
        img = self.ensure_channels(img)

        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk).type(torch.int64)

        # Returning to iterator.
        return img, msk, img_filename

    def __len__(self):
        return len(self.items)
