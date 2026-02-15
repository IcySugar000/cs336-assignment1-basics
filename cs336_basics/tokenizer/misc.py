import time
import json
from typing import Literal

import numpy as np
from loguru import logger

from cs336_basics.tokenizer.trainer import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


def tokenize_dataset(dataset: Literal["TinyStories", "owt"], num_chunks: int = 64):
    vocab_path = f"checkpoints/tokenizer/{dataset}_vocab.json"
    merges_path = f"checkpoints/tokenizer/{dataset}_merges.txt"
    special_tokens = ["<|endoftext|>"] if dataset == "TinyStories" else None
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    splitter = "<|endoftext|>" if dataset == "TinyStories" else "\n"

    results = np.array([], dtype=np.uint16)

    for dataset_type in ["train", "valid"]:
        with open(f"data/{dataset}_{dataset_type}.txt", "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, splitter.encode())

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                logger.info(f"Tokenizing file: {start} - {end} / {boundaries[-1]}")
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                new_data = np.array(tokenizer.encode(chunk), dtype=np.uint16)
                results = np.concatenate([results, new_data])
        np.save(f"checkpoints/tokenizer/{dataset}_{dataset_type}_encoded.npy", results)


def scalene_tokenizer():
    tokenizer = Tokenizer.from_files(
        "checkpoints/tokenizer/TinyStories_vocab.json",
        "checkpoints/tokenizer/TinyStories_merges.txt",
        ["<|endoftext|>"],
    )
    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        data = f.read()
        tokenizer.encode(data)


def speed_test():
    tokenizer = Tokenizer.from_files(
        "checkpoints/tokenizer/TinyStories_vocab.json",
        "checkpoints/tokenizer/TinyStories_merges.txt",
        ["<|endoftext|>"],
    )
    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        data = f.read()
        start = time.perf_counter()
        tokenizer.encode(data)
        time_used = time.perf_counter() - start
        byte_len = len(data.encode())
        logger.info(f"Speed: {byte_len / time_used} bytes / second")


if __name__ == "__main__":
    tokenize_dataset("TinyStories")
    # scalene_tokenizer()
    # speed_test()
