import einops as eo
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

from src.data.classification.base import _DataModule, _Dataset


class SeedRGB(_Dataset):
    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        img_path = self.get_image_path(index)

        # C, H, W
        img = np.asarray(Image.open(img_path))
        img = eo.rearrange(torch.from_numpy(img / 255.0), "h w c -> c h w")

        if self.transform:
            img = self.transform(img)

        if self.multi_task:
            return (
                img,
                self.labels[index],
                self.genus_labels[index],
                self.family_labels[index],
            )
        return img, self.labels[index]


class SeedRGBModule(_DataModule):
    def setup(self, stage: str) -> None:
        self.ds_train = SeedRGB(
            self.path_train,
            base_dirs=self.base_dirs,
            base_dirs_sampling=self.base_dirs_sampling[0],
            transform=v2.Compose(self.train_transform),
            max_size=self.max_size,
        )
        self.ds_val = SeedRGB(
            self.path_val,
            base_dirs=self.base_dirs,
            base_dirs_sampling=self.base_dirs_sampling[1],
            transform=v2.Compose(self.val_transform),
            max_size=self.max_size,
        )
        self.ds_test = SeedRGB(
            self.path_test,
            base_dirs=self.base_dirs,
            base_dirs_sampling=self.base_dirs_sampling[2],
            transform=v2.Compose(self.test_transform),
            max_size=self.max_size,
        )

        self.class_names = self.class_names or self.ds_train.class_names
