import torch
from einops import einsum, rearrange


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        thetas = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        pos = torch.arange(0, max_seq_len, device=device)
        angles = einsum(thetas, pos, "d, seq -> seq d")

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

        self.cos: torch.Tensor
        self.sin: torch.Tensor

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        x_0, x_1 = x[..., 0::2], x[..., 1::2]
        x_0_r = x_0 * cos - x_1 * sin
        x_1_r = x_0 * sin + x_1 * cos
        return rearrange([x_0_r, x_1_r], "two ... d -> ... (d two)")
