import torch
from einops import rearrange, einsum

from .rope import RoPE
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

        self.w_q = torch.nn.Parameter(torch.rand(num_heads * self.d_k, d_model, device=device))
        self.w_k = torch.nn.Parameter(torch.rand(num_heads * self.d_k, d_model, device=device))
        self.w_v = torch.nn.Parameter(torch.rand(num_heads * self.d_v, d_model, device=device))
        self.w_o = torch.nn.Parameter(torch.rand(d_model, num_heads * self.d_v, device=device))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        w_q = rearrange(self.w_q, "(head dk) dm -> head dk dm", head=self.num_heads)
        w_k = rearrange(self.w_k, "(head dk) dm -> head dk dm", head=self.num_heads)
        w_v = rearrange(self.w_v, "(head dv) dm -> head dv dm", head=self.num_heads)
        q = einsum(w_q, x, "head dk dm, ... seq dm -> ... head seq dk")
        k = einsum(w_k, x, "head dk dm, ... seq dm -> ... head seq dk")
        v = einsum(w_v, x, "head dv dm, ... seq dm -> ... head seq dv")

        if self.rope is not None and token_positions is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        mask = torch.tril(torch.ones(x.size(-2), x.size(-2), dtype=torch.bool))
        heads = scaled_dot_product_attention(q, k, v, mask)
        multihead = rearrange(heads, "... head seq d_v -> ... seq (head d_v)")
        return einsum(self.w_o, multihead, "d_model hd_v, ... seq hd_v -> ... seq d_model")
