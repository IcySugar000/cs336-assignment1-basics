import torch
from einops import rearrange

from .rope import RoPE
from .linear import Linear
from cs336_basics.utils import scaled_dot_product_attention


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.qkv = Linear(d_model, 3 * (num_heads * self.d_k), device=device)
        self.o = Linear(num_heads * self.d_v, d_model, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        qkv_projected = self.qkv.forward(x)
        q, k, v = rearrange(
            qkv_projected, "... seq (three head dk) -> three ... head seq dk", three=3, head=self.num_heads
        )

        if self.rope is not None and token_positions is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        mask = torch.tril(torch.ones(x.size(-2), x.size(-2), dtype=torch.bool, device=q.device))
        heads = scaled_dot_product_attention(q, k, v, mask)
        multihead = rearrange(heads, "... head seq d_v -> ... seq (head d_v)")
        return self.o.forward(multihead)
