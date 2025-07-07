import copy
import warnings

import torch.optim as optim


class StepLrWithWarmup(optim.lr_scheduler.StepLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        step_size: int,
        warmup_step: int,
        keep_step: int = 0,  # 0 means no keep
        gamma=0.1,
        last_epoch=-1,
    ) -> None:
        self.warmup_step = warmup_step
        self.keep_step = keep_step
        self.init_lr_groups = copy.deepcopy(optimizer.param_groups)
        super().__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self._step_count <= self.warmup_step:  # type: ignore
            lr_scale = min(1.0, float(self._step_count) / self.warmup_step)  # type: ignore
            lr = [group["lr"] * lr_scale for group in self.init_lr_groups]
            return lr
        if (
            self.keep_step == 0
            or self._step_count - self.warmup_step <= self.keep_step
            or (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
        ):  # type: ignore
            lr = [group["lr"] for group in self.optimizer.param_groups]  # type: ignore
            return lr
        lr = [group["lr"] * self.gamma for group in self.optimizer.param_groups]  # type: ignore
        return lr


class CosineWarmupLR(optim.lr_scheduler.SequentialLR):
    """
    cosine_iters: number of iterations for cosine annealing
    keep_iters: number of iterations to keep the learning rate after warmup
    warmup_factor: factor to warmup the learning rate
    end_factor: factor to end the learning rate at the end of warmup
    warmup_iters: number of iterations for warmup
    eta_min: minimum learning rate after cosine annealing
    eta_min_factor: factor to multiply the minimum learning rate, if provided,
        eta_min = eta_min_factor * initial_lr
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        cosine_iters: int,
        keep_iters: int,
        warmup_factor: float = 0.1,
        end_factor: float = 1,
        warmup_iters: int = 5,
        eta_min: float = 0.0,
        eta_min_factor: float | None = None,
    ) -> None:
        initial_lr = optimizer.param_groups[0]["lr"]
        if eta_min > initial_lr:
            warnings.warn(
                f"{eta_min=} is greater than the initial learning rate = {initial_lr}. "
                "This will result in eta_min being set to 0 !"
            )
            eta_min = 0.0
        if eta_min_factor and isinstance(eta_min_factor, float):
            eta_min = initial_lr * eta_min_factor

        schedulers = [
            optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=warmup_factor,
                end_factor=end_factor,
                total_iters=warmup_iters,
            ),
            optim.lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=1.0,
                total_iters=keep_iters,
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cosine_iters,
                eta_min=eta_min,
            ),
            optim.lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=eta_min / initial_lr,
                total_iters=int(1e6),
            ),
        ]
        super().__init__(
            optimizer,
            schedulers,
            milestones=[
                warmup_iters,
                warmup_iters + keep_iters,
                warmup_iters + keep_iters + cosine_iters,
            ],
        )
