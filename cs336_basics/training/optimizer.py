import math
from collections.abc import Callable, Iterator

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        weight_decay=0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, betas, eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
            b1, b2 = betas
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * (grad**2)
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    for lr in [1, 1e1, 1e2, 1e3]:
        print(f"lr: {lr}")
        opt = SGD([weights], lr=lr)
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
