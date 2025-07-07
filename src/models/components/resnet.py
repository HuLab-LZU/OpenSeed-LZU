from typing import Callable, Dict, Tuple

import torch.optim as optim
from timm.models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152

from src.models.components.common import SeedModelBase

ARCHS = {
    "resnet_18": resnet18,
    "resnet_34": resnet34,
    "resnet_50": resnet50,
    "resnet_101": resnet101,
    "resnet_152": resnet152,
}


class SeedResNet(SeedModelBase):
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
        func: Callable[..., ResNet] = ARCHS[arch]
        self.model = func(
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
