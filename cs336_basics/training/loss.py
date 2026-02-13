import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_values, _ = torch.max(inputs, dim=-1, keepdim=True)
    o = inputs - max_values
    loss = torch.sum(torch.log(torch.sum(torch.exp(o), dim=-1))) - torch.sum(o[torch.arange(targets.size(0)), targets])
    loss /= targets.size(0)
    return loss


def perplexity(losses: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.mean(losses))
