from typing import Tuple

import einops as eo
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch import Tensor
from PIL import Image

from src.data.classification.base import _DataModule, _Dataset


class SeedRGB(_Dataset):
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img_path = self.get_image_path(index)

        # C, H, W
        img = np.asarray(Image.open(img_path))
        # img: Any = cv.cvtColor(cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        img = eo.rearrange(torch.from_numpy(img / 255.0), "h w c -> c h w")

        # h, w = img.size
        # if h > self.max_size[0] or w > self.max_size[1]:
        #     img = self.resize_image(img)

        if self.transform:
            img = self.transform(img)
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


if __name__ == "__main__":
    ...
