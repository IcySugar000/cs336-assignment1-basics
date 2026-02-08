import torch

from .attention import MultiheadSelfAttention
from .swiglu import SwiGLU
from .norm import RMSNorm
from .rope import RoPE


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.device = device
        if rope is not None:
            self.rope = rope
        elif theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        else:
            self.rope = None
        self.mha = MultiheadSelfAttention(d_model, num_heads, device, rope=self.rope)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.size(-2), device=self.device)
        x = x + self.mha.forward(self.norm1.forward(x), token_positions)
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x
