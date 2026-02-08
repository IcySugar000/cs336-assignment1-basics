import torch
from einops import reduce, einsum


class RMSNorm(torch.nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = reduce(x**2, "... d_model -> ... 1", "mean")
        rms = torch.sqrt(rms + self.eps)

        result = einsum(x / rms, self.g, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)
