import json
from collections.abc import Callable, Sequence
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as tm
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torch import nn, optim

from src.models.components.losses import FocalLoss, build_class_weights
from src.utils import RankedLogger

ev_logger = RankedLogger(__name__, rank_zero_only=True)


def _as_float(x: torch.Tensor | float) -> float:
    if torch.is_tensor(x):
        return x.item()
    return float(x)


class SeedModelBase(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[..., optim.Optimizer],
        scheduler: Callable[..., optim.lr_scheduler.LRScheduler] | None = None,
        in_channels: int = 3,
        num_classes: int = 656,
        img_size: tuple[int, int] = (224, 224),
        scheduler_args: dict | None = None,
        compile: bool = False,
        loss_type: str = "none",
        loss_gamma: float = 2.0,
        loss_beta: float = 0.9999,
        loss_alpha: float | None = None,
        num_classes_family: int | None = None,
        num_classes_genus: int | None = None,
        loss_weight_species: float = 1.0,
        loss_weight_genus: float = 0.5,
        loss_weight_family: float = 0.25,
        consistency_weight: float = 0.1,
    ):
        super().__init__()

        self.optimizer_func = optimizer
        self.scheduler_func = scheduler
        self.num_classes = num_classes
        self.scheduler_args = scheduler_args or {}

        # Multi-task (family / genus / species) support.
        self.num_classes_family = num_classes_family
        self.num_classes_genus = num_classes_genus
        self.multi_task = num_classes_family is not None and num_classes_genus is not None
        self.head_species: nn.Module | None = None
        self.head_genus: nn.Module | None = None
        self.head_family: nn.Module | None = None

        # Hierarchical loss configuration.
        # Defaults reflect the fact that species is the primary, most specific
        # task and also the most long-tailed; coarser genus/family act as
        # regularizers and should receive progressively smaller weight.
        self.loss_weight_species = loss_weight_species
        self.loss_weight_genus = loss_weight_genus
        self.loss_weight_family = loss_weight_family
        self.consistency_weight = consistency_weight
        self._species_to_genus: torch.Tensor | None = None
        self._genus_to_family: torch.Tensor | None = None

        self.train_acc = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.train_acc_genus = (
            tm.Accuracy(
                task="multiclass",
                num_classes=num_classes_genus,
                average="macro",
            )
            if self.multi_task
            else None
        )
        self.train_acc_family = (
            tm.Accuracy(
                task="multiclass",
                num_classes=num_classes_family,
                average="macro",
            )
            if self.multi_task
            else None
        )

        self.val_acc = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="none",
        )
        self.val_acc_3 = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            top_k=3,
        )
        self.val_acc_genus = (
            tm.Accuracy(
                task="multiclass",
                num_classes=num_classes_genus,
                average="none",
            )
            if self.multi_task
            else None
        )
        self.val_acc_family = (
            tm.Accuracy(
                task="multiclass",
                num_classes=num_classes_family,
                average="none",
            )
            if self.multi_task
            else None
        )

        # Per-class test metrics. ``average="none"`` returns one value per class,
        # so macro averaging can be restricted to classes that actually appear
        # in the test split (test set may legitimately contain a subset of the
        # training classes, e.g. after a plate-disjoint split).
        self.test_acc = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="none",
        )
        self.test_acc_3 = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            top_k=3,
        )
        self.test_cm = tm.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )
        self.test_acc_genus = (
            tm.Accuracy(
                task="multiclass",
                num_classes=num_classes_genus,
                average="none",
            )
            if self.multi_task
            else None
        )
        self.test_acc_family = (
            tm.Accuracy(
                task="multiclass",
                num_classes=num_classes_family,
                average="none",
            )
            if self.multi_task
            else None
        )
        self.test_y_true: list[int] = []
        self.test_y_pred: list[int] = []
        self.test_y_pred_subset: list[int] = []
        self.test_y_true_genus: list[int] = []
        self.test_y_pred_genus: list[int] = []
        self.test_y_pred_genus_subset: list[int] = []
        self.test_y_true_family: list[int] = []
        self.test_y_pred_family: list[int] = []
        self.test_y_pred_family_subset: list[int] = []

        self.test_present_mask_species: torch.Tensor | None = None
        self.test_present_mask_genus: torch.Tensor | None = None
        self.test_present_mask_family: torch.Tensor | None = None

        self._val_batches_seen: int = 0

        self.is_compile = compile
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_beta = loss_beta
        self.loss_alpha = loss_alpha
        self._class_counts: torch.Tensor | None = None
        self._class_weights: torch.Tensor | None = None

        self.cm = tm.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )

        self.val_acc_best = tm.MaxMetric()

        # batch size of 8N is required for fp8 training
        self.example_input_array = torch.randn(8, in_channels, img_size[0], img_size[1])

    # ------------------------------------------------------------------ #
    # Multi-task helpers
    # ------------------------------------------------------------------ #
    def _init_multitask_heads(self) -> None:
        """Build family/genus/species heads from the backbone feature map.

        Must be called by subclasses after ``self.model`` is assigned.
        """
        if not self.multi_task:
            return
        feat_dim = getattr(self.model, "num_features", None)
        if feat_dim is None:
            raise AttributeError(
                f"{type(self.model).__name__} has no `num_features`; multi-task heads require a timm-style backbone."
            )
        self.head_species = nn.Linear(feat_dim, self.num_classes)
        self.head_genus = nn.Linear(feat_dim, self.num_classes_genus)  # type: ignore
        self.head_family = nn.Linear(feat_dim, self.num_classes_family)  # type: ignore

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool backbone feature maps to a per-sample vector."""
        if x.ndim == 3:
            # (B, tokens, C) e.g. ViT; average over tokens
            return x.mean(dim=1)
        if x.ndim != 4:
            return x
        feat_dim = getattr(self.model, "num_features", None)
        if feat_dim is not None and x.shape[-1] == feat_dim:
            # (B, H, W, C) e.g. Swin
            return x.mean(dim=(1, 2))
        # (B, C, H, W) e.g. CNN
        return x.mean(dim=(2, 3))

    def _num_val_batches(self) -> int | None:
        """Expected number of validation batches for the (single) val loader."""
        n = getattr(self.trainer, "num_val_batches", None)
        if n is None:
            return None
        if isinstance(n, (list, tuple)):
            return int(n[0]) if n else None
        return int(n)

    def _present_mask(self, labels: list[int] | None, num_classes: int) -> torch.Tensor:
        if labels is None:
            return torch.ones(num_classes, dtype=torch.bool)
        counts = torch.bincount(
            torch.tensor(labels, dtype=torch.long),
            minlength=num_classes,
        )
        return counts > 0

    def _restricted_softmax(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return F.softmax(logits, dim=-1)
        mask = mask.to(logits.device)
        return F.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)

    def _ensure_hierarchy_mappings(self) -> None:
        """Build species->genus and genus->family incidence matrices.

        Uses the training dataset (falls back to test) to derive the
        deterministic per-species taxonomy mapping.  Called once lazily.
        """
        if not self.multi_task or (self._species_to_genus is not None and self._genus_to_family is not None):
            return
        dm = getattr(self.trainer, "datamodule", None)
        ds = getattr(dm, "ds_train", None) if dm is not None else None
        if ds is None:
            ds = getattr(dm, "ds_test", None) if dm is not None else None
        if ds is None or not getattr(ds, "multi_task", False):
            return
        labels = getattr(ds, "labels", None)
        genus_labels = getattr(ds, "genus_labels", None)
        family_labels = getattr(ds, "family_labels", None)
        if not labels:
            return

        sp_to_gen = torch.zeros(
            self.num_classes,
            self.num_classes_genus,  # type: ignore
            dtype=torch.long,
        )
        gen_to_fam = torch.zeros(
            self.num_classes_genus,  # type: ignore
            self.num_classes_family,  # type: ignore
            dtype=torch.long,
        )
        seen_species: set[int] = set()
        seen_genus: set[int] = set()
        for sp, ge, fa in zip(labels, genus_labels, family_labels):  # type: ignore
            if sp not in seen_species:
                sp_to_gen[sp, ge] = 1
                seen_species.add(sp)
            if ge not in seen_genus:
                gen_to_fam[ge, fa] = 1
                seen_genus.add(ge)
        self._species_to_genus = sp_to_gen.float()
        self._genus_to_family = gen_to_fam.float()

    def _hierarchical_consistency_loss(
        self,
        sp_logits: torch.Tensor,
        ge_logits: torch.Tensor,
        fa_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise disagreement between the genus/family heads and the
        species-derived taxonomic marginals.

        The species head is treated as the primary predictor, so its marginal
        distribution is detached.  The genus head is regularised toward the
        species->genus marginal; the family head is regularised toward the
        genus->family marginal.
        """
        if self.consistency_weight <= 0 or not self.multi_task:
            return torch.tensor(0.0, device=sp_logits.device)
        self._ensure_hierarchy_mappings()
        if self._species_to_genus is None or self._genus_to_family is None:
            return torch.tensor(0.0, device=sp_logits.device)
        device = sp_logits.device
        sp_probs = F.softmax(sp_logits, dim=-1).detach()
        ge_probs = F.softmax(ge_logits, dim=-1)
        sp_to_gen = self._species_to_genus.to(device)
        gen_to_fam = self._genus_to_family.to(device)

        genus_marginal = torch.matmul(sp_probs, sp_to_gen)  # (B, n_genus)
        family_marginal = torch.matmul(ge_probs.detach(), gen_to_fam)  # (B, n_family)

        kl_genus = F.kl_div(F.log_softmax(ge_logits, dim=-1), genus_marginal, reduction="batchmean")
        kl_family = F.kl_div(F.log_softmax(fa_logits, dim=-1), family_marginal, reduction="batchmean")
        return self.consistency_weight * (kl_genus + kl_family)

    # ------------------------------------------------------------------ #
    # Lightning lifecycle
    # ------------------------------------------------------------------ #
    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.is_compile and stage == "fit":
            self.model = torch.compile(self.model)  # type: ignore

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.multi_task:
            feats = self.model.forward_features(x)  # type: ignore
            feats = self._pool_features(feats)
            sp = self.head_species(feats)  # type: ignore
            ge = self.head_genus(feats)  # type: ignore
            fa = self.head_family(feats)  # type: ignore
            return sp, ge, fa  # type: ignore
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_acc.reset()
        self.val_acc_3.reset()
        if self.val_acc_genus is not None:
            self.val_acc_genus.reset()
        if self.val_acc_family is not None:
            self.val_acc_family.reset()
        self.cm.reset()
        self.val_acc_best.reset()
        self._ensure_class_counts()
        self._ensure_hierarchy_mappings()

    def _ensure_class_counts(self) -> None:
        """Compute per-class counts from the training dataset once."""
        if self._class_counts is not None or self.loss_type == "none":
            return
        dm = getattr(self.trainer, "datamodule", None)
        labels = None
        if dm is not None:
            ds = getattr(dm, "ds_train", None)
            if ds is not None:
                labels = getattr(ds, "labels", None)
        if labels is None:
            ev_logger.warning("class counts not available; loss weighting disabled")
            self._class_counts = torch.ones(self.num_classes)
        else:
            counts = torch.bincount(
                torch.tensor(labels, dtype=torch.long),
                minlength=self.num_classes,
            ).float()
            self._class_counts = counts
        if self.loss_type in ("inverse_freq", "class_balanced"):
            self._class_weights = build_class_weights(
                self._class_counts,
                mode=self.loss_type,
                beta=self.loss_beta,
            )
        else:
            self._class_weights = None

    def _training_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the configured training loss."""
        if self.loss_type == "none":
            return F.cross_entropy(logits, y)
        if self.loss_type in ("inverse_freq", "class_balanced"):
            self._ensure_class_counts()
            if self._class_weights is None:
                return F.cross_entropy(logits, y)
            weights = self._class_weights.to(logits.device)
            return F.cross_entropy(logits, y, weight=weights)
        if self.loss_type == "focal":
            alpha = self.loss_alpha
            if alpha is not None and not isinstance(alpha, torch.Tensor):
                alpha = float(alpha)
            return FocalLoss(gamma=self.loss_gamma, alpha=alpha)(logits, y)
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _mt_loss(
        self,
        sp_logits: torch.Tensor,
        ge_logits: torch.Tensor,
        fa_logits: torch.Tensor,
        y_species: torch.Tensor,
        y_genus: torch.Tensor,
        y_family: torch.Tensor,
    ) -> torch.Tensor:
        loss = (
            self.loss_weight_species * self._training_loss(sp_logits, y_species)
            + self.loss_weight_genus * F.cross_entropy(ge_logits, y_genus)
            + self.loss_weight_family * F.cross_entropy(fa_logits, y_family)
        )
        consistency = self._hierarchical_consistency_loss(
            sp_logits,
            ge_logits,
            fa_logits,
        )
        return loss + consistency

    def training_step(self, batch, batch_idx):
        if self.multi_task:
            x, y_species, y_genus, y_family = batch
            sp_logits, ge_logits, fa_logits = self(x)
            loss = self._mt_loss(
                sp_logits,
                ge_logits,
                fa_logits,
                y_species,
                y_genus,
                y_family,
            )
            self.train_acc(F.softmax(sp_logits, dim=-1), y_species)
            self.train_acc_genus(F.softmax(ge_logits, dim=-1), y_genus)  # type: ignore
            self.train_acc_family(F.softmax(fa_logits, dim=-1), y_family)  # type: ignore
            self.log("train_loss", loss)
            self.log("train_acc", self.train_acc)
            self.log("train_acc_genus", self.train_acc_genus)  # type: ignore
            self.log("train_acc_family", self.train_acc_family)  # type: ignore
            return loss

        x, y = batch
        logits: torch.Tensor = self(x)
        loss = self._training_loss(logits, y)
        probs = F.softmax(logits, dim=-1)
        self.train_acc(probs, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        return loss

    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        if self.multi_task:
            x, y_species, y_genus, y_family = batch
            sp_logits, ge_logits, fa_logits = self(x)
            loss = self._mt_loss(
                sp_logits,
                ge_logits,
                fa_logits,
                y_species,
                y_genus,
                y_family,
            )
            self.val_acc(F.softmax(sp_logits, dim=-1), y_species)
            self.val_acc_3(F.softmax(sp_logits, dim=-1), y_species)
            self.val_acc_genus(F.softmax(ge_logits, dim=-1), y_genus)  # type: ignore
            self.val_acc_family(F.softmax(fa_logits, dim=-1), y_family)  # type: ignore
            self.cm(sp_logits.argmax(dim=-1), y_species)
            self._val_batches_seen += 1
            self.log("val_loss", loss, prog_bar=True)
            return loss

        x, y = batch
        logits: torch.Tensor = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)  # (B, num_classes)
        self.val_acc(probs, y)
        self.val_acc_3(probs, y)
        self.cm(probs.argmax(dim=-1), y)
        self._val_batches_seen += 1
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset per-epoch validation state.

        Without this reset, ``val_acc``/`val_acc_3`/`cm` accumulate scores from
        every previous validation epoch, so the logged ``val_acc`` becomes a
        cumulative average over all past epochs and systematically understates
        the current epoch's accuracy.  It also makes the confusion-matrix
        ``present`` mask and macro metrics stale.
        """
        self._val_batches_seen = 0
        self.val_acc.reset()
        self.val_acc_3.reset()
        if self.val_acc_genus is not None:
            self.val_acc_genus.reset()
        if self.val_acc_family is not None:
            self.val_acc_family.reset()
        self.cm.reset()

    def on_validation_epoch_end(self) -> None:
        expected = self._num_val_batches()
        if expected is not None and self._val_batches_seen < expected:
            ev_logger.warning(
                f"Validation incomplete: saw {self._val_batches_seen}/{expected} batches; "
                "skipping val_acc logging/checkpointing and confmat write."
            )
            return

        self.log_confusion_matrix(confmat_name=f"confmat_val_{self.global_step}")

        cm: torch.Tensor = self.cm.confmat  # type: ignore
        support = cm.sum(dim=1)
        present = support > 0

        per_class_acc = self.val_acc.compute()  # (num_classes,), 0 for absent classes
        per_class_acc_3 = self.val_acc_3.compute()

        val_acc_micro = cm.diag().sum() / support.sum()
        val_acc_macro_present = per_class_acc[present].mean()
        val_acc_macro_full = per_class_acc.mean()

        val_acc_top3_micro = (per_class_acc_3 * support).sum() / support.sum()
        val_acc_top3_macro_present = per_class_acc_3[present].mean()
        val_acc_top3_macro_full = per_class_acc_3.mean()

        self.log("val_acc", val_acc_macro_present, prog_bar=True)
        self.log("val_acc_macro_full", val_acc_macro_full)
        self.log("val_acc_micro", val_acc_micro)
        self.log("val_acc_top3", val_acc_top3_macro_present)
        self.log("val_acc_top3_micro", val_acc_top3_micro)
        self.log("val_acc_top3_macro_full", val_acc_top3_macro_full)

        dm = getattr(self.trainer, "datamodule", None)
        ds_val = getattr(dm, "ds_val", None) if dm is not None else None
        if self.val_acc_genus is not None:
            ga = self.val_acc_genus.compute()
            genus_labels = getattr(ds_val, "genus_labels", None) if ds_val is not None else None
            gmask = self._present_mask(genus_labels, self.num_classes_genus).to(ga.device)  # type: ignore
            self.log("val_acc_genus", ga[gmask].mean() if gmask.any() else 0)
        if self.val_acc_family is not None:
            fa = self.val_acc_family.compute()
            family_labels = getattr(ds_val, "family_labels", None) if ds_val is not None else None
            fmask = self._present_mask(family_labels, self.num_classes_family).to(fa.device)  # type: ignore
            self.log("val_acc_family", fa[fmask].mean() if fmask.any() else 0)

        self.val_acc_best(val_acc_macro_present)  # update best so far val acc
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)

    def on_test_start(self) -> None:
        """Reset test-only metrics so validation state never leaks into test logs."""
        self.test_acc.reset()
        self.test_acc_3.reset()
        self.test_cm.reset()
        if self.test_acc_genus is not None:
            self.test_acc_genus.reset()
        if self.test_acc_family is not None:
            self.test_acc_family.reset()
        self.test_y_true = []
        self.test_y_pred = []
        self.test_y_pred_subset = []
        self.test_y_true_genus = []
        self.test_y_pred_genus = []
        self.test_y_pred_genus_subset = []
        self.test_y_true_family = []
        self.test_y_pred_family = []
        self.test_y_pred_family_subset = []

        dm = getattr(self.trainer, "datamodule", None)
        ds = getattr(dm, "ds_test", None) if dm is not None else None
        labels = getattr(ds, "labels", None) if ds is not None else None
        self.test_present_mask_species = self._present_mask(labels, self.num_classes)
        if self.multi_task:
            genus_labels = getattr(ds, "genus_labels", None) if ds is not None else None
            family_labels = getattr(ds, "family_labels", None) if ds is not None else None
            self.test_present_mask_genus = self._present_mask(genus_labels, self.num_classes_genus)  # type: ignore
            self.test_present_mask_family = self._present_mask(family_labels, self.num_classes_family)  # type: ignore
        self._ensure_hierarchy_mappings()

    def test_step(self, batch, batch_idx):
        if self.multi_task:
            x, y_species, y_genus, y_family = batch
            sp_logits, ge_logits, fa_logits = self(x)
            loss = self._mt_loss(sp_logits, ge_logits, fa_logits, y_species, y_genus, y_family)
            probs = F.softmax(sp_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            probs_subset = self._restricted_softmax(sp_logits, self.test_present_mask_species)
            preds_subset = probs_subset.argmax(dim=-1)

            probs_genus = F.softmax(ge_logits, dim=-1)
            probs_genus_subset = self._restricted_softmax(ge_logits, self.test_present_mask_genus)
            probs_family = F.softmax(fa_logits, dim=-1)
            probs_family_subset = self._restricted_softmax(fa_logits, self.test_present_mask_family)

            self.test_acc(probs, y_species)
            self.test_acc_3(probs, y_species)
            self.test_cm(preds, y_species)
            self.test_acc_genus(probs_genus, y_genus)  # type: ignore
            self.test_acc_family(probs_family, y_family)  # type: ignore

            self.test_y_true.extend(y_species.cpu().tolist())
            self.test_y_pred.extend(preds.cpu().tolist())
            self.test_y_pred_subset.extend(preds_subset.cpu().tolist())
            self.test_y_true_genus.extend(y_genus.cpu().tolist())
            self.test_y_pred_genus.extend(probs_genus.argmax(dim=-1).cpu().tolist())
            self.test_y_pred_genus_subset.extend(probs_genus_subset.argmax(dim=-1).cpu().tolist())
            self.test_y_true_family.extend(y_family.cpu().tolist())
            self.test_y_pred_family.extend(probs_family.argmax(dim=-1).cpu().tolist())
            self.test_y_pred_family_subset.extend(probs_family_subset.argmax(dim=-1).cpu().tolist())
            self.log("test_loss", loss)
            return loss

        x, y = batch
        logits: torch.Tensor = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        probs_subset = self._restricted_softmax(logits, self.test_present_mask_species)
        preds_subset = probs_subset.argmax(dim=-1)

        self.test_acc(probs, y)
        self.test_acc_3(probs, y)
        self.test_cm(preds, y)
        self.test_y_true.extend(y.cpu().tolist())
        self.test_y_pred.extend(preds.cpu().tolist())
        self.test_y_pred_subset.extend(preds_subset.cpu().tolist())
        self.log("test_loss", loss)
        return loss

    def _macro_from_lists(self, y_true: list[int], y_pred: list[int], num_classes: int) -> tuple[float, float, float]:
        """Return (micro, macro_present, macro_full) for integer label/pred lists."""
        if not y_true:
            return float("nan"), float("nan"), float("nan")
        yt = torch.tensor(y_true, dtype=torch.long)
        yp = torch.tensor(y_pred, dtype=torch.long)
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        # Advanced indexing `cm[yt, yp] += 1` does NOT accumulate duplicate
        # (y_true, y_pred) pairs; use `index_put_` with accumulate=True instead.
        cm.index_put_((yt, yp), torch.ones_like(yt), accumulate=True)
        support = cm.sum(dim=1)
        present = support > 0
        micro = cm.diag().sum() / support.sum()
        per = cm.diag() / support.clamp(min=1)
        macro_present = per[present].mean() if present.any() else 0.0
        macro_full = per.mean()
        return float(micro), float(macro_present), float(macro_full)

    def on_test_epoch_end(self) -> None:
        cm: torch.Tensor = self.test_cm.confmat  # type: ignore
        support = cm.sum(dim=1)
        present = support > 0

        per_class_acc = self.test_acc.compute()  # (num_classes,), 0 for absent classes
        per_class_acc_3 = self.test_acc_3.compute()

        micro_acc = cm.diag().sum() / support.sum()
        macro_present_acc = per_class_acc[present].mean()
        macro_full_acc = per_class_acc.mean()  # absent classes count as 0

        micro_acc_3 = (per_class_acc_3 * support).sum() / support.sum()
        macro_present_acc_3 = per_class_acc_3[present].mean()
        macro_full_acc_3 = per_class_acc_3.mean()

        # Restricted candidate set (test-present species only).
        (
            micro_subset,
            macro_present_subset,
            _,
        ) = self._macro_from_lists(self.test_y_true, self.test_y_pred_subset, self.num_classes)

        self.log("test_acc_micro", micro_acc, prog_bar=True)
        self.log("test_acc_macro_present", macro_present_acc, prog_bar=True)
        self.log("test_acc_macro_full", macro_full_acc)
        self.log("test_acc_top3_micro", micro_acc_3)
        self.log("test_acc_top3_macro_present", macro_present_acc_3)
        self.log("test_acc_top3_macro_full", macro_full_acc_3)
        self.log("test_acc_micro_subset", micro_subset)
        self.log("test_acc_macro_subset", macro_present_subset)

        # Multi-task hierarchical metrics.
        genus_metrics: dict[str, float] = {}
        family_metrics: dict[str, float] = {}
        if self.multi_task:
            (
                genus_micro,
                genus_macro_present,
                genus_macro_full,
            ) = self._macro_from_lists(self.test_y_true_genus, self.test_y_pred_genus, self.num_classes_genus)  # type: ignore
            (
                genus_micro_subset,
                genus_macro_subset,
                _,
            ) = self._macro_from_lists(
                self.test_y_true_genus,
                self.test_y_pred_genus_subset,
                self.num_classes_genus,  # type: ignore
            )
            (
                family_micro,
                family_macro_present,
                family_macro_full,
            ) = self._macro_from_lists(
                self.test_y_true_family,
                self.test_y_pred_family,
                self.num_classes_family,  # type: ignore
            )
            (
                family_micro_subset,
                family_macro_subset,
                _,
            ) = self._macro_from_lists(
                self.test_y_true_family,
                self.test_y_pred_family_subset,
                self.num_classes_family,  # type: ignore
            )
            genus_metrics = {
                "test_acc_genus_micro": genus_micro,
                "test_acc_genus_macro_present": genus_macro_present,
                "test_acc_genus_macro_full": genus_macro_full,
                "test_acc_genus_micro_subset": genus_micro_subset,
                "test_acc_genus_macro_subset": genus_macro_subset,
            }
            family_metrics = {
                "test_acc_family_micro": family_micro,
                "test_acc_family_macro_present": family_macro_present,
                "test_acc_family_macro_full": family_macro_full,
                "test_acc_family_micro_subset": family_micro_subset,
                "test_acc_family_macro_subset": family_macro_subset,
            }
            for k, v in {**genus_metrics, **family_metrics}.items():
                self.log(k, v)

        if self.trainer.is_global_zero:
            out_dir = Path(self.trainer.default_root_dir or self.trainer.log_dir or ".")
            out_dir.mkdir(parents=True, exist_ok=True)

            test_metrics = {
                "test_acc_micro": _as_float(micro_acc),
                "test_acc_macro_present": _as_float(macro_present_acc),
                "test_acc_macro_full": _as_float(macro_full_acc),
                "test_acc_top3_micro": _as_float(micro_acc_3),
                "test_acc_top3_macro_present": _as_float(macro_present_acc_3),
                "test_acc_top3_macro_full": _as_float(macro_full_acc_3),
                "test_acc_micro_subset": _as_float(micro_subset),
                "test_acc_macro_subset": _as_float(macro_present_subset),
                "test_classes_present": int(present.sum()),
                "num_classes": self.num_classes,
                "num_samples": len(self.test_y_true),
                **genus_metrics,
                **family_metrics,
            }
            with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            eval_test: dict = {
                "species": {
                    "y_true": self.test_y_true,
                    "y_pred": self.test_y_pred,
                    "y_pred_subset": self.test_y_pred_subset,
                }
            }
            if self.multi_task:
                eval_test["genus"] = {
                    "y_true": self.test_y_true_genus,
                    "y_pred": self.test_y_pred_genus,
                    "y_pred_subset": self.test_y_pred_genus_subset,
                }
                eval_test["family"] = {
                    "y_true": self.test_y_true_family,
                    "y_pred": self.test_y_pred_family,
                    "y_pred_subset": self.test_y_pred_family_subset,
                }
            with open(out_dir / "eval_test.json", "w", encoding="utf-8") as f:
                json.dump(eval_test, f, ensure_ascii=False)

            self.log_confusion_matrix(
                cm=self.test_cm,
                confmat_name="confmat_test",
                out_dir=out_dir,
            )

        ev_logger.info(
            f"Test results saved to {self.trainer.default_root_dir or self.trainer.log_dir}; "
            f"test classes present: {int(present.sum())}/{self.num_classes}"
        )

    def log_confusion_matrix(
        self,
        class_names: Sequence[str] | None = None,
        confmat_name: str | None = None,
        cm=None,
        out_dir: Path | str | None = None,
    ):
        if cm is None:
            cm = self.cm
        confmat: torch.Tensor = cm.confmat.cpu().detach().numpy()  # type: ignore
        fields = class_names or [f"{i}" for i in range(self.num_classes)]
        out_dir = (
            Path(out_dir) if out_dir is not None else Path(self.trainer.log_dir or self.trainer.default_root_dir or ".")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_dir / f"{confmat_name}.csv",
            confmat,
            fmt="%d",
            delimiter=",",
            header=",".join(fields),
        )

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
