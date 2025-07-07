from typing import Callable, Dict, Tuple

import torch.optim as optim
import torch.nn as nn
from timm.models.mobilenetv3 import (
    mobilenetv4_conv_medium,
    mobilenetv4_conv_small,
    mobilenetv4_hybrid_medium,
    mobilenetv4_conv_large,
    mobilenetv4_hybrid_large,
)

from src.models.components.common import SeedModelBase

ARCHS = {
    "mobilenet_conv_small": mobilenetv4_conv_small,
    "mobilenet_conv_medium": mobilenetv4_conv_medium,
    "mobilenet_hybrid_medium": mobilenetv4_hybrid_medium,
    "mobilenet_conv_large": mobilenetv4_conv_large,
    "mobilenet_hybrid_large": mobilenetv4_hybrid_large,
}


class SeedMobileNet(SeedModelBase):
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
