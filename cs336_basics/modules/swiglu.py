import torch
from einops import einsum


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None):
        super().__init__()
        if not d_ff:
            d_ff = round(d_model * 8 / 3 / 64) * 64
        self.w1_weight = torch.nn.Parameter(torch.rand(d_ff, d_model, device=device))
        self.w2_weight = torch.nn.Parameter(torch.rand(d_model, d_ff, device=device))
        self.w3_weight = torch.nn.Parameter(torch.rand(d_ff, d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        w3x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        result = self._silu(w1x) * w3x
        result = einsum(self.w2_weight, result, "d_model d_ff, ... d_ff -> ... d_model")
        return result

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, torch.sigmoid(x), "... d, ... d -> ... d")
