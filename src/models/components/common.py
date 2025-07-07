from typing import Callable, Dict, Sequence, Tuple

from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig, OptimizerConfig

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm

from src.utils import RankedLogger

ev_logger = RankedLogger(__name__, rank_zero_only=True)


class SeedModelBase(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[..., optim.Optimizer],
        scheduler: Callable[..., optim.lr_scheduler.LRScheduler] | None = None,
        in_channels: int = 3,
        num_classes: int = 656,
        img_size: Tuple[int, int] = (224, 224),
        scheduler_args: Dict | None = None,
        compile: bool = False,
    ):
        super().__init__()

        self.optimizer_func = optimizer
        self.scheduler_func = scheduler
        self.num_classes = num_classes
        self.scheduler_args = scheduler_args or {}

        self.train_acc = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )

        self.val_acc = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.val_acc_3 = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            top_k=3,
        )

        self.is_compile = compile

        # self.val_precision = tm.Precision(
        #     task="multiclass",
        #     num_classes=num_classes,
        #     average="macro",
        # )

        # self.val_recall = tm.Recall(
        #     task="multiclass",
        #     num_classes=num_classes,
        #     average="macro",
        # )

        # self.val_f1 = tm.F1Score(
        #     task="multiclass",
        #     num_classes=num_classes,
        #     average="macro",
        # )

        # self.val_pr_curve = tm.PrecisionRecallCurve(
        #     task="multiclass",
        #     num_classes=num_classes,
        #     average="macro",
        # )

        # # self.val_roc = tm.ROc

        self.cm = tm.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )

        self.val_acc_best = tm.MaxMetric()

        # batch size of 8N is required for fp8 training
        self.example_input_array = torch.randn(8, in_channels, img_size[0], img_size[1])

    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.get("compile", None) and stage == "fit":
        if self.is_compile and stage == "fit":
            self.model = torch.compile(self.model)  # type: ignore

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch, batch_idx):
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch
        logits: torch.Tensor = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)

        self.train_acc(probs, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        return loss

    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()
        logger = self.logger.experiment  # type: ignore
        for name, param in self.named_parameters():
            if name.startswith("model"):
                logger.add_histogram(name, param, self.global_step)
                if param.requires_grad and param.grad is not None:
                    logger.add_histogram(f"{name}_grad", param.grad, self.global_step)

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch
        logits: torch.Tensor = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)  # (B, num_classes)
        self.val_acc(probs, y)
        self.val_acc_3(probs, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_acc_top3", self.val_acc_3)

        self.cm(probs.argmax(dim=-1), y)
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.log_confusion_matrix(confmat_name=f"confmat_val_{self.global_step}")
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch
        logits: torch.Tensor = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)
        self.val_acc(probs, y)
        self.val_acc_3(probs, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.val_acc)
        self.log("test_acc_top3", self.val_acc_3)

        self.cm(probs.argmax(dim=-1), y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_confusion_matrix(confmat_name="confmat_test")
        ev_logger.info(f"Confusion matrix saved to {self.trainer.log_dir}/confmat_test.csv")

    def log_confusion_matrix(
        self,
        class_names: Sequence[str] | None = None,
        confmat_name: str | None = None,
    ):
        confmat: torch.Tensor = self.cm.confmat.cpu().detach().numpy()  # type: ignore
        fields = class_names or [f"{i}" for i in range(self.num_classes)]
        np.savetxt(
            f"{self.trainer.log_dir}/{confmat_name}.csv",
            confmat,
            fmt="%d",
            delimiter=",",
            header=",".join(fields),
        )

        # table = PrettyTable(fields)
        # for i, row in enumerate(confmat):
        #     table.add_row([f"{i}", *[f"{j:.0f}" for j in row]])
        # print(self.trainer.log_dir)
        # writer: SummaryWriter = self.logger.experiment  # type: ignore
        # writer.add_text("test_confusion", table.get_html_string(), self.global_step)  # type: ignore
        # table.get_csv_string()
        # fig = sns.heatmap(
        #     confmat.cpu().detach().numpy(),
        #     annot=False,
        #     cmap="Blues",
        # ).get_figure()
        # fig.tight_layout()  # type: ignore
        ...

    def configure_optimizers(self) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        assert self.trainer.model is not None, "Model not found!"
        optimizer = self.optimizer_func(params=self.trainer.model.parameters())
        if self.scheduler_func is None:
            return {"optimizer": optimizer}

        scheduler = self.scheduler_func(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.scheduler_args.get("monitor", None) or "val_loss",
                "interval": self.scheduler_args.get("interval", None) or "epoch",
                "frequency": self.scheduler_args.get("frequency", None) or 1,
            },
        }

    # def get_progress_bar_dict(self):
    #     tqdm_dict = super().get_progress_bar_dict()
    #     tqdm_dict.pop("v_num", None)
    #     return tqdm_dict
