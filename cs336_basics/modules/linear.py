import math

import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.w = torch.nn.Parameter(torch.ones(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.w, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
