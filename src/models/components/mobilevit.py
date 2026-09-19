from collections.abc import Callable

from timm.models.mobilevit import (
    mobilevitv2_050,
    # mobilevitv2_075,
    mobilevitv2_100,
    # mobilevitv2_125,
    mobilevitv2_150,
    # mobilevitv2_175,
    # mobilevitv2_200,
)
from torch import nn, optim

from src.models.components.common import SeedModelBase

ARCHS = {
    "mobilevit_050": mobilevitv2_050,
    # "mobilevit_075": mobilevitv2_075,
    "mobilevit_100": mobilevitv2_100,
    # "mobilevit_125": mobilevitv2_125,
    "mobilevit_150": mobilevitv2_150,
    # "mobilevit_175": mobilevitv2_175,
    # "mobilevit_200": mobilevitv2_200,
}


class SeedMobileViT(SeedModelBase):
    def __init__(
        self,
        arch: str,
        optimizer: Callable[..., optim.Optimizer],
        scheduler: Callable[..., optim.lr_scheduler.LRScheduler] | None = None,
        in_channels: int = 3,
        num_classes: int = 656,
        num_classes_family: int | None = None,
        num_classes_genus: int | None = None,
        img_size: tuple[int, int] = (224, 224),
        scheduler_args: dict | None = None,
        pretrained: bool = False,
        compile: bool = True,
        loss_type: str = "none",
        loss_gamma: float = 2.0,
        loss_beta: float = 0.9999,
        loss_alpha: float | None = None,
        loss_weight_species: float = 1.0,
        loss_weight_genus: float = 0.5,
        loss_weight_family: float = 0.25,
        consistency_weight: float = 0.1,
    ):
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            in_channels=in_channels,
            num_classes=num_classes,
            num_classes_family=num_classes_family,
            num_classes_genus=num_classes_genus,
            img_size=img_size,
            scheduler_args=scheduler_args,
            compile=compile,
            loss_type=loss_type,
            loss_gamma=loss_gamma,
            loss_beta=loss_beta,
            loss_alpha=loss_alpha,
            loss_weight_species=loss_weight_species,
            loss_weight_genus=loss_weight_genus,
            loss_weight_family=loss_weight_family,
            consistency_weight=consistency_weight,
        )
        func: Callable[..., nn.Module] = ARCHS[arch]
        self.model: nn.Module = func(
            pretrained=pretrained,
            num_classes=0 if self.multi_task else num_classes,
            in_chans=in_channels,
        )
        if self.multi_task:
            self._init_multitask_heads()
