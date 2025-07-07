from typing import Callable, Dict, Tuple

import torch.optim as optim
import torch.nn as nn
from timm.models.mobilevit import (
    mobilevitv2_050,
    mobilevitv2_075,
    mobilevitv2_100,
    mobilevitv2_125,
    mobilevitv2_150,
    mobilevitv2_175,
    mobilevitv2_200,
)

from src.models.components.common import SeedModelBase

ARCHS = {
    "mobilevit_050": mobilevitv2_050,
    "mobilevit_075": mobilevitv2_075,
    "mobilevit_100": mobilevitv2_100,
    "mobilevit_125": mobilevitv2_125,
    "mobilevit_150": mobilevitv2_150,
    "mobilevit_175": mobilevitv2_175,
    "mobilevit_200": mobilevitv2_200,
}


class SeedMobileViT(SeedModelBase):
    def __init__(
        self,
        arch: str,
        optimizer: Callable[..., optim.Optimizer],
        scheduler: Callable[..., optim.lr_scheduler.LRScheduler] | None = None,
        in_channels: int = 3,
        num_classes: int = 656,
        img_size: Tuple[int, int] = (224, 224),
        scheduler_args: Dict | None = None,
        pretrained: bool = False,
        compile: bool = True,
    ):
        super().__init__(
            optimizer,
            scheduler,
            in_channels,
            num_classes,
            img_size,
            scheduler_args=scheduler_args,
            compile=compile,
        )
        func: Callable[..., nn.Module] = ARCHS[arch]
        self.model: nn.Module = func(
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
