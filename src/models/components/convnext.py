from typing import Callable, Dict, Tuple

import torch.optim as optim
from timm.models.convnext import (
    ConvNeXt,
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
)

from src.models.components.common import SeedModelBase

ARCHS = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
}


class SeedConvNeXt(SeedModelBase):
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
            optimizer=optimizer,
            scheduler=scheduler,
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            scheduler_args=scheduler_args,
            compile=compile,
        )

        func: Callable[..., ConvNeXt] = ARCHS[arch]
        self.model: ConvNeXt = func(
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
