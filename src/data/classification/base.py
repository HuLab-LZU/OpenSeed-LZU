import json
import os
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Literal, TypeAlias

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

BDSType: TypeAlias = Literal["all", "uniform"] | list[float]


class BaseDirSamplingMethod(Enum):
    DISABLE = "all"
    PROBABILISTIC = "probabilistic"


class RandomGamma(nn.Module):
    """Apply a random gamma correction to a float tensor in [0, 1]."""

    def __init__(self, min_gamma: float = 0.8, max_gamma: float = 1.25) -> None:
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        gamma = float(torch.empty(1).uniform_(self.min_gamma, self.max_gamma).item())
        return torch.pow(torch.clamp(x, 0.0, 1.0), gamma)


class RandomChannelGain(nn.Module):
    """Randomly scale each color channel independently (simulates white-balance)."""

    def __init__(self, min_gain: float = 0.9, max_gain: float = 1.1) -> None:
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if x.ndim == 3:
            scale = torch.empty(x.shape[0], 1, 1).uniform_(self.min_gain, self.max_gain).to(x.device)
        elif x.ndim == 4:
            scale = torch.empty(x.shape[0], x.shape[1], 1, 1).uniform_(self.min_gain, self.max_gain).to(x.device)
        else:
            raise ValueError(f"Unsupported tensor dims: {x.ndim}")
        return torch.clamp(x * scale, 0.0, 1.0)


# classification
class _Dataset(Dataset):
    def __init__(
        self,
        ds_path: str,  # "*.json"
        base_dirs: list[str] | str | None = None,  # {base_dir}/1.png
        base_dirs_sampling: BDSType = "all",
        transform: nn.Module | None = None,
        max_size: tuple[int, int] = (256, 256),
    ) -> None:
        assert os.path.exists(ds_path) and ds_path.endswith(".json")
        """
            _ds: {
                "version": "",
                "author": "",
                "images": <str>[],
                "labels": <int>[],
                "classes": <str>[],
            }
        """
        with open(ds_path, "r", encoding="utf-8") as f:
            _ds = json.load(f)
        images: list[str] = _ds["images"]
        labels: list[int] = _ds["labels"]
        classes = _ds["classes"]

        # Multi-task JSON: classes is a dict {family: [...], genus: [...], species: [...]}
        # and labels_family / labels_genus are parallel to labels.
        self.multi_task = isinstance(classes, dict) and "species" in classes
        if self.multi_task:
            self.class_names = classes["species"]
            self.num_classes_species = len(classes["species"])
            self.num_classes_genus = len(classes.get("genus", []))
            self.num_classes_family = len(classes.get("family", []))
            self.genus_labels: list[int] = _ds.get("labels_genus", [])
            self.family_labels: list[int] = _ds.get("labels_family", [])
        else:
            self.class_names = classes
            self.num_classes_species = len(self.class_names)
            self.genus_labels = None
            self.family_labels = None

        self.transform = transform
        self.max_size = max_size
        self.sampling_method = (
            BaseDirSamplingMethod.DISABLE if base_dirs_sampling == "all" else BaseDirSamplingMethod.PROBABILISTIC
        )

        # final used in __getitem__
        self.img_names: list[str] = []  # names if sampling else full path
        self.labels: list[int] = []
        self.base_dirs: list[str] = []
        self.base_dirs_sampling_p: list[float] = []

        if base_dirs is None:
            self.img_names = images
            self.labels = labels
            self.base_dirs = [""]  # necessary for os.path.join()
        else:
            assert isinstance(base_dirs, (list, str)), (
                f"base_dirs must be a list of str or a str, but got {type(base_dirs)}"
            )
            # set self.base_dirs
            if isinstance(base_dirs, str):
                self.base_dirs = [base_dirs]
            else:
                self.base_dirs = base_dirs

            # process sampling
            self.img_names = images
            self.labels = labels
            if base_dirs_sampling == "all":
                self.img_names = []
                self.labels = labels * len(self.base_dirs)
                if self.multi_task:
                    self.genus_labels = self.genus_labels * len(self.base_dirs)
                    self.family_labels = self.family_labels * len(self.base_dirs)
                for d in self.base_dirs:
                    self.img_names += [os.path.join(d, name) for name in images]
            elif base_dirs_sampling == "uniform":
                n = len(self.base_dirs)
                self.base_dirs_sampling_p = [1 / n for _ in range(n)]
            elif all(isinstance(p, float) for p in base_dirs_sampling):
                assert len(base_dirs_sampling) == len(self.base_dirs), (
                    f"base_dirs_sampling must have the same length as base_dirs, but got {len(base_dirs_sampling)} and {len(self.base_dirs)}"
                )
                self.base_dirs_sampling_p = base_dirs_sampling
            else:
                raise ValueError(
                    f"base_dirs_sampling must be 'all', 'uniform' or a list of float, but got {base_dirs_sampling}"
                )

        self._len = len(self.labels)

    def get_image_path(self, idx: int) -> str:
        # will be the path of image if base_dirs_sampling == "all"
        # else, the image name needed to be joined with base_dir
        img_path = self.img_names[idx]
        if self.base_dirs_sampling_p and self.sampling_method == BaseDirSamplingMethod.PROBABILISTIC:
            base_dir = np.random.choice(self.base_dirs, p=self.base_dirs_sampling_p)
            img_path = os.path.join(base_dir, img_path)
        return img_path

    def resize_image(self, img: Image.Image) -> Image.Image:
        h, w = img.size
        fac = min(self.max_size[0] / h, self.max_size[1] / w)
        newsize = (int(h * fac), int(w * fac))
        # img = F.interpolate(img, size=newsize)
        img = img.resize(newsize, resample=Image.Resampling.BILINEAR)
        return img

    def __len__(self) -> int:
        return self._len


class _DataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train: str,
        path_val: str,
        path_test: str,
        base_dirs: list[str],
        base_dirs_sampling: tuple[BDSType, BDSType, BDSType] = (
            "uniform",
            "all",
            "all",
        ),
        data_dir: str | None = None,
        batch_size: int = 16,
        max_size: tuple[int, int] = (256, 256),
        num_workers: int = 4,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        norm_mean: list[float] | None = None,
        norm_std: list[float] | None = None,
        class_names: Sequence[str] | None = None,
        pin_memory: bool = False,
        prompt_drop_rate: float = 0.5,
        prompt_elem_drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.path_train = path_train if data_dir is None else os.path.join(data_dir, path_train)
        self.path_val = path_val if data_dir is None else os.path.join(data_dir, path_val)
        self.path_test = path_test if data_dir is None else os.path.join(data_dir, path_test)

        self.base_dirs: list[str]
        if data_dir is None or base_dirs is None:
            self.base_dirs = base_dirs
        else:
            self.base_dirs = [os.path.join(data_dir, d) for d in base_dirs]
        self.base_dirs_sampling = base_dirs_sampling

        assert os.path.exists(self.path_train) and path_train.endswith(".json")
        assert os.path.exists(self.path_val) and path_val.endswith(".json")
        assert os.path.exists(self.path_test) and path_test.endswith(".json")
        assert all(os.path.exists(d) for d in self.base_dirs)

        self.batch_size = batch_size
        self.max_size = max_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.ds_train: _Dataset = None  # type: ignore
        self.ds_val: _Dataset = None  # type: ignore
        self.ds_test: _Dataset = None  # type: ignore
        self.class_names: Sequence[str] | None = class_names

        self.pin_memory = pin_memory
        self.prompt_drop_rate = prompt_drop_rate
        self.prompt_elem_drop_rate = prompt_elem_drop_rate

    @property
    def train_transform(self) -> list[Callable | nn.Module]:
        return [
            v2.ToDtype(torch.float32),
            v2.RandomRotation((0, 360)),
            v2.RandomResizedCrop(
                size=self.max_size,
                scale=(0.8, 1.0),
                ratio=(0.8, 2.0),
            ),
            # photometric augmentation to reduce sensitivity to illumination.
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            v2.RandomApply([RandomGamma(0.8, 1.25)], p=0.3),
            v2.RandomApply([RandomChannelGain(0.9, 1.1)], p=0.3),
            # Background/imprint random erasing.
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ] + self.post_transform

    @property
    def val_transform(self) -> list[Callable | nn.Module]:
        return [
            v2.ToDtype(torch.float32),
            v2.Resize(self.max_size),
        ] + self.post_transform

    @property
    def post_transform(self) -> list[Callable | nn.Module]:
        if self.norm_mean and self.norm_std:
            return [
                v2.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        return []

    @property
    def test_transform(self) -> list[Callable | nn.Module]:
        return self.val_transform

    def prepare_data(self) -> None: ...

    def train_dataloader(self):
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        assert self.ds_test is not None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
