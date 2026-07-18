"""Small PyTorch implementation of the NATTEN 0.14.6 1-D module used by PLUTO.

PLUTO imports only :class:`NeighborhoodAttention1D`.  The original NATTEN
0.14.6 binary has no Windows wheel, so this module reproduces its forward
definition with ordinary differentiable PyTorch operations.  Parameter names
and shapes intentionally match NATTEN 0.14.6, allowing strict checkpoint
loading.  It is optimized for PLUTO's short history sequences, not as a
general replacement for NATTEN's CUDA kernels.

Reference implementation:
https://github.com/SHI-Labs/NATTEN/tree/v0.14.6
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

__version__ = "0.14.6-pytorch-fallback"


def _window_and_bias_indices(
    length: int, kernel_size: int, dilation: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return NATTEN-compatible key and relative-position-bias indices."""

    neighborhood = kernel_size // 2
    key_rows: list[list[int]] = []
    bias_rows: list[list[int]] = []

    for index in range(length):
        if dilation <= 1:
            start = max(index - neighborhood, 0)
            if index + neighborhood >= length:
                start += length - index - neighborhood - 1
            bias_start = neighborhood
            if index < neighborhood:
                bias_start += neighborhood - index
            if index + neighborhood >= length:
                bias_start += length - index - 1 - neighborhood
        else:
            start = index - neighborhood * dilation
            if start < 0:
                start = index % dilation
            elif index + neighborhood * dilation >= length:
                imod = index % dilation
                aligned = (length // dilation) * dilation
                remainder = length - aligned
                if imod < remainder:
                    start = length - remainder + imod - 2 * neighborhood * dilation
                else:
                    start = aligned + imod - kernel_size * dilation

            if index - neighborhood * dilation < 0:
                bias_start = kernel_size - 1 - index // dilation
            elif index + neighborhood * dilation >= length:
                bias_start = (length - index - 1) // dilation
            else:
                bias_start = neighborhood

        key_rows.append([start + offset * dilation for offset in range(kernel_size)])
        bias_rows.append([bias_start + offset for offset in range(kernel_size)])

    return (
        torch.tensor(key_rows, dtype=torch.long, device=device),
        torch.tensor(bias_rows, dtype=torch.long, device=device),
    )


class NeighborhoodAttention1D(nn.Module):
    """API- and state-dict-compatible subset of NATTEN 0.14.6."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int | None = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be an odd integer > 1, got {kernel_size}")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if dilation is not None and dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(torch.zeros(num_heads, 2 * kernel_size - 1))
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch, original_length, channels = x.shape
        length = original_length
        if length < self.window_size:
            x = F.pad(x, (0, 0, 0, self.window_size - length))
            length = self.window_size

        qkv = (
            self.qkv(x)
            .reshape(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query = query * self.scale

        key_index, bias_index = _window_and_bias_indices(
            length, self.kernel_size, self.dilation, x.device
        )
        # [B, H, L, K, D], gathered without materializing a full LxL matrix.
        local_key = key[:, :, key_index, :]
        logits = torch.einsum("bhld,bhlkd->bhlk", query, local_key)
        if self.rpb is not None:
            logits = logits + self.rpb[:, bias_index].unsqueeze(0)

        attention = self.attn_drop(logits.softmax(dim=-1))
        local_value = value[:, :, key_index, :]
        output = torch.einsum("bhlk,bhlkd->bhld", attention, local_value)
        output = output.permute(0, 2, 1, 3).reshape(batch, length, channels)
        output = output[:, :original_length, :]
        return self.proj_drop(self.proj(output))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            f"rel_pos_bias={self.rpb is not None}"
        )


__all__ = ["NeighborhoodAttention1D", "_window_and_bias_indices"]
