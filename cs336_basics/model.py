import torch

from .modules import TransformerBlock, Embedding, RMSNorm, Linear, RoPE


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length)
        self.embedding = Embedding(vocab_size, d_model, device)
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        x = self.linear.forward(x)
        return x
