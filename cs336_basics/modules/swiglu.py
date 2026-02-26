import torch
from einops import einsum

from .linear import Linear


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None):
        super().__init__()
        if not d_ff:
            d_ff = round(d_model * 8 / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1.forward(x)
        w3x = self.w3.forward(x)
        result = self.w2.forward(self._silu(w1x) * w3x)
        return result

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, torch.sigmoid(x), "... d, ... d -> ... d")
