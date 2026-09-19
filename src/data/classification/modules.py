import torch
import torch.nn.functional as F
from torch import nn


class ResizePad(nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size  # h, w
        self.mh, self.mw = size

    def forward(self, x: torch.Tensor):
        _, h, w = x.shape
        if w > self.mw or h > self.mh:
            ratio = min(self.mh / h, self.mw / w)
            newsize = (int(h * ratio), int(w * ratio))
            x = F.interpolate(x.unsqueeze(0), size=newsize, mode="bilinear", align_corners=False).squeeze(0)
        _, h, w = x.shape
        # pad to `self.size`
        if w < self.mw or h < self.mh:
            x = F.pad(
                x,
                pad=[0, max(0, self.mw - w), 0, max(0, self.mh - h)],
                mode="constant",
                value=0,
            )
        return x

    def __repr__(self):
        return f"ResizePad(size={self.size})"


class Resize(nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = (size[0], size[1])  # h, w

    def forward(self, x: torch.Tensor):
        x = F.interpolate(
            x.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return x

    def __repr__(self):
        return f"Resize(size={self.size})"


class MinMaxNormalize(torch.nn.Module):
    """
    Normalize a tensor to [0, 1] by min-max normalization.
    input: torch.Tensor, shape=(C, H, W)
    output: torch.Tensor, shape=(C, H, W)
    Will keep `C` dimension and reduce `H` and `W` dimension.
    """

    def __init__(self, global_reduce: bool = False) -> None:
        super().__init__()
        self.global_reduce = global_reduce

    def forward(self, x: torch.Tensor):
        # x: C, H, W
        if self.global_reduce:
            return (x - x.min()) / (x.max() - x.min())
        _min, _ = x.min(dim=-1, keepdim=True)[0].min(dim=-1, keepdim=True)
        _max, _ = x.max(dim=-1, keepdim=True)[0].max(dim=-1, keepdim=True)
        return (x - _min) / (_max - _min)

    def __repr__(self):
        return f"MinMaxNormalize(dim={self.dim})"
