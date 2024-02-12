from abc import ABC

from data.base_dataset import BaseDataset
from data.typings import SimpleDataTuple


class SimpleDataset(BaseDataset, ABC):
    def __getitem__(self, index: int) -> SimpleDataTuple:
        img, msk, img_filename = self.get_data(index)

        img = self.prepare_image_as_tensor(img)
        msk = self.prepare_mask_as_tensor(msk)

        # Returning to iterator.
        return SimpleDataTuple(
            image=img, mask=msk, file_name=img_filename, dataset_name=self.dataset_name
        )

    def __len__(self):
        return len(self.items)
