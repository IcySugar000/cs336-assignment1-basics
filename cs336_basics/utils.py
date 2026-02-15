import os
import math
import random
import typing
from functools import lru_cache
from collections.abc import Iterable

import torch
import numpy as np
from einops import einsum


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def softmax(x: torch.Tensor, dim: int, temp: float = 1.0):
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    x_sub = x - max_values
    exp_x = torch.exp(x_sub / temp)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    result = einsum(q, k, "batch ... query d_k, batch ... key d_k -> batch ... query key")
    result /= math.sqrt(q.size(-1))
    if mask is not None:
        result = torch.where(mask, result, float("-inf"))
    result = softmax(result, -1)
    result = einsum(result, v, "batch ... query key, batch ... key d_v -> batch ... query d_v")
    return result


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    l2 = math.sqrt(sum([param.grad.norm(p=2) ** 2 for param in parameters if param.grad is not None]))
    if l2 > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm / (l2 + eps)


def get_batch(dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str):
    inputs = torch.empty((batch_size, context_length), dtype=torch.int, device=device)
    targets = torch.empty_like(inputs)
    idxs = [random.randint(0, dataset.size - context_length - 1) for _ in range(batch_size)]
    for i, idx in enumerate(idxs):
        inputs[i] = torch.from_numpy(dataset[idx : idx + context_length])
        targets[i] = torch.from_numpy(dataset[idx + 1 : idx + context_length + 1])
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(data, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    data = torch.load(src)
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return data["iteration"]
