import os
import time
import random
import typing

import torch
import numpy as np
from loguru import logger
from einops import rearrange

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.training import AdamW, cross_entropy, CosineScheduleParams, perplexity
from cs336_basics.utils import save_checkpoint, load_checkpoint, get_batch, softmax


def training_loop(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    batch_size: int,
    steps: int,
    train_data_path: str | os.PathLike,
    validation_data_path: str | os.PathLike,
    lr: float = 1e-3,
    lr_scheduling_params: CosineScheduleParams | None = None,
    weight_decay: float = 0.01,
    opti_betas: tuple[float, float] = (0.9, 0.999),
    opti_eps: float = 1e-8,
    load_from: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] | None = None,
    save_to: str = "checkpoints/model",
    device: str = "cpu",
):
    # 1. Init
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=torch.device(device),
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=opti_betas,
        eps=opti_eps,
        scheduling=lr_scheduling_params,
    )

    iteration = 1
    if load_from:
        iteration = load_checkpoint(src=load_from, model=model, optimizer=optimizer) + 1
    optimizer.set_iter_from(iteration)

    training_data = np.load(train_data_path, "r")
    validation_data = np.load(validation_data_path, "r")

    # 2. Training
    model.train()
    for step in range(iteration, steps + 1):
        start = time.perf_counter()
        batch_inputs, batch_targets = get_batch(training_data, batch_size, context_length, device)
        batch_actuals = model.forward(batch_inputs)
        batch_actuals = rearrange(batch_actuals, "batch seq vocab -> (batch seq) vocab")
        batch_targets = rearrange(batch_targets, "batch seq -> (batch seq)")
        losses = cross_entropy(batch_actuals, batch_targets)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        time_used = time.perf_counter() - start
        logger.info(f"Step {step} training loss: {losses.item()}, time used: {time_used:.3f}")

        if step % 100 == 0 or step == steps:
            # Validation
            model.eval()
            validation_inputs, validation_targets = get_batch(validation_data, batch_size, context_length, device)
            validation_targets = rearrange(validation_targets, "batch seq -> (batch seq)")
            with torch.no_grad():
                validation_actuals = model.forward(validation_inputs)
                validation_actuals = rearrange(validation_actuals, "batch seq vocab -> (batch seq) vocab")
                losses = cross_entropy(validation_actuals, validation_targets)
                logger.info(f"Validation loss: {losses.item()}, perplexity: {perplexity(losses)}")
            model.train()

            # Save
            filepath = f"{save_to}/{step}.cp"
            filedir = os.path.dirname(filepath)
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            save_checkpoint(model, optimizer, step, filepath)


def generate(
    prompt: str,
    model: TransformerLM,
    max_length: int,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None,
    stop_token: str = "<|endoftext|>",
    temp: float = 1.0,
    top_p: float = 0.0,
    device: str | None = None,
) -> str:
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    tokens_list = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens_list, dtype=torch.int, device=device)
    for _ in range(max_length):
        logits = model.forward(tokens)
        prob = softmax(logits, 0, temp)

        prob_list: list[float] = prob.tolist()
        prob_id: dict[int, float] = {i: p for i, p in enumerate(prob_list)}
        prob_sorted = sorted(prob_id.items(), key=lambda x: x[1])
        prob_sum = 0.0
        cleaned_prob, cleand_id = [], []
        while prob_sum < top_p:
            i, p = prob_sorted.pop()
            cleaned_prob.append(p)
            cleand_id.append(i)
            prob_sum += p
        cleaned_prob = [p / prob_sum for p in cleaned_prob]
        next_token_id = random.choices(cleand_id, weights=cleaned_prob)[0]
        tokens = torch.cat([tokens, torch.tensor([next_token_id], tokens.dtype, tokens.device)])
        if tokenizer.vocab[next_token_id].decode() == stop_token:
            break

    tokens_id: list[int] = tokens.tolist()
    result = tokenizer.decode(tokens_id)
    return result


if __name__ == "__main__":
    training_loop(
        10000,
        256,
        4,
        512,
        16,
        1344,
        10000,
        32,
        5000,
        "checkpoints/tokenizer/TinyStories_train_encoded.npy",
        "checkpoints/tokenizer/TinyStories_valid_encoded.npy",
    )
