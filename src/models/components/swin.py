from typing import Callable, Dict, Tuple

import torch.optim as optim
from timm.models.swin_transformer import (
    SwinTransformer,
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
    swin_large_patch4_window7_224,
)

from src.models.components.common import SeedModelBase

ARCHS = {
    "swin_tiny": swin_tiny_patch4_window7_224,
    "swin_small": swin_small_patch4_window7_224,
    "swin_base": swin_base_patch4_window7_224,
    "swin_large": swin_large_patch4_window7_224,
}


class SeedSwin(SeedModelBase):
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

        func: Callable[..., SwinTransformer] = ARCHS[arch]
        self.model = func(
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
