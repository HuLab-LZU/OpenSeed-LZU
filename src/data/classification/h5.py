from typing import Any, Callable, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torch import Tensor

from src.data.classification.modules import NormalizeH5, Resize, MinMaxNormalize
from src.data.classification.base import _DataModule, _Dataset


class SeedH5(_Dataset):
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img_path = self.get_image_path(index)
        # C, H, W
        fin = h5py.File(img_path, "r")
        img: Any = fin["pixel_values"]
        img = torch.from_numpy(np.asarray(img, dtype=np.float32))
        fin.close()
        if self.transform:
            img = self.transform(img)

        return img, self.labels[index]


class SeedH5Module(_DataModule):
    @property
    def post_transform(self) -> List[Callable | nn.Module]:
        if self.norm_mean and self.norm_std:
            return [
                NormalizeH5(
                    mean=self.norm_mean,
                    std=self.norm_std,
                ),
            ]
        return [MinMaxNormalize(global_reduce=False)]

    def setup(self, stage: str) -> None:
        self.ds_train = SeedH5(
            self.path_train,
            base_dirs=self.base_dirs,
            base_dirs_sampling=self.base_dirs_sampling[0],
            transform=v2.Compose([Resize(self.max_size)] + self.post_transform),
            max_size=self.max_size,
        )
        self.ds_val = SeedH5(
            self.path_val,
            base_dirs=self.base_dirs,
            base_dirs_sampling=self.base_dirs_sampling[1],
            transform=v2.Compose([Resize(self.max_size)] + self.post_transform),
            max_size=self.max_size,
        )
        self.ds_test = SeedH5(
            self.path_test,
            base_dirs=self.base_dirs,
            base_dirs_sampling=self.base_dirs_sampling[2],
            transform=v2.Compose([Resize(self.max_size)] + self.post_transform),
            max_size=self.max_size,
        )

        self.class_names = self.class_names or self.ds_train.class_names
