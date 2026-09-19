import tempfile
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torchvision
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from torch.utils.data import DataLoader


class ImageSamplerCallback(Callback):
    def __init__(
        self,
        padding: int = 2,
        normalize: bool = False,
        norm_range: tuple[int, int] | None = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value


class TensorboardImageSampler(ImageSamplerCallback):
    """Generates images and logs to tensorboard. Your model must implement the ``forward`` function for generation.

    Requirements::

        # model must have img_dim arg
        model.img_dim = (1, 28, 28)

        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)

    Example::

        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images: list[torch.Tensor]
        loader: DataLoader = trainer.train_dataloader  # type: ignore
        _, images = next(enumerate(loader))  # type: ignore
        assert loader.batch_size is not None
        nrow = int(sqrt(loader.batch_size))

        grid = torchvision.utils.make_grid(
            tensor=images[0],
            nrow=nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)  # type: ignore


class WandbImageSampler(ImageSamplerCallback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images: list[torch.Tensor]
        loader: DataLoader = trainer.train_dataloader  # type: ignore
        _, images = next(enumerate(loader))  # type: ignore
        assert loader.batch_size is not None
        nrow = int(sqrt(loader.batch_size))

        grid = torchvision.utils.make_grid(
            tensor=images[0],
            nrow=nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        logger: WandbLogger = trainer.logger  # type: ignore
        logger.log_image(str_title, [grid], step=trainer.global_step)  # type: ignore


class MlFlowImageSampler(ImageSamplerCallback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images: list[torch.Tensor]
        loader: DataLoader = trainer.train_dataloader  # type: ignore
        _, images = next(enumerate(loader))  # type: ignore
        assert loader.batch_size is not None
        nrow = int(sqrt(loader.batch_size))

        grid = (
            torchvision.utils
            .make_grid(
                tensor=images[0],
                nrow=nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )
            .permute(1, 2, 0)
            .cpu()
            .detach()
            .numpy()
        )
        grid = (grid - grid.min()) / (grid.max() - grid.min())
        grid *= 255
        grid = grid.astype(np.uint8)
        str_title = f"{pl_module.__class__.__name__}_images"
        logger: MLFlowLogger = trainer.logger  # type: ignore
        logger.experiment.log_image(
            logger.run_id,
            image=grid,
            key=str_title,
            step=trainer.global_step,
        )  # type: ignore


# https://github.com/Lightning-AI/pytorch-lightning/issues/16310#issuecomment-2241008059
class MLFlowSaveConfigCallback(SaveConfigCallback):
    def __init__(
        self,
        parser,
        config,
        config_filename="config.yaml",
        overwrite=False,
        multifile=False,
    ):
        super().__init__(
            parser,
            config,
            config_filename,
            overwrite,
            multifile,
            save_to_log_dir=False,
        )

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # convert namespace to dict
        # config_dict = vars(self.config)

        if trainer.is_global_zero:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.yaml"
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                trainer.logger.experiment.log_artifact(  # type: ignore
                    local_path=config_path,
                    run_id=trainer.logger.run_id,  # type: ignore
                )
