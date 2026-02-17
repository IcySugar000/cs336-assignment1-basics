import os
import json
import heapq
import multiprocessing
from typing import BinaryIO, cast
from functools import partial

import regex as re
from loguru import logger

from cs336_basics.utils import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class MaxHeapItem:
    def __init__(self, val: tuple[tuple[bytes, bytes], int]):
        self.val = val

    def __lt__(self, other: "MaxHeapItem"):
        if self.val[1] != other.val[1]:
            return self.val[1] > other.val[1]
        if self.val[0][0] != other.val[0][0]:
            return self.val[0][0] > other.val[0][0]
        return self.val[0][1] >= other.val[0][1]

    def __eq__(self, other: "MaxHeapItem"):
        return self.val == other.val

    def __repr__(self):
        return f"MaxHeapItem({self.val!r})"


class Trainer:
    vocab: dict[int, bytes]
    vocab_size: int
    vocab_values: set[bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    words: dict[tuple[bytes, ...], int]
    token_to_words: dict[bytes, set[tuple[bytes, ...]]]

    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        splitter: str = "<|endoftext|>",
        num_splits: int = 4,
        num_processes: int = 4,
    ):
        # 1. 词表设置与默认填充
        self.vocab_size = vocab_size
        self.vocab = {i: i.to_bytes() for i in range(256)}
        for i in range(len(special_tokens)):
            self.vocab[256 + i] = special_tokens[i].encode()
        self.vocab_values = set(self.vocab.values())

        # 2. 初始化成员变量
        self.merges = []
        self.special_tokens = special_tokens
        self.words = {}
        self.token_to_words = {}

        # 3. 并行切分文件内容，进行pretokenize
        def chunk_generator():
            with open(input_path, "rb") as f:
                boundaries = find_chunk_boundaries(f, num_splits, splitter.encode())

                # The following is a serial implementation, but you can parallelize this
                # by sending each start/end pair to a set of processes.
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    logger.info(f"Sending pretokenizing file: {start} - {end} / {boundaries[-1]}")
                    f.seek(start)
                    yield f.read(end - start).decode("utf-8", errors="ignore")

        if num_processes > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                pretokenize_func = partial(self.pretokenize, special_tokens=special_tokens)
                for i, result in enumerate(pool.imap_unordered(pretokenize_func, chunk_generator())):
                    self.merge_words(result)
                    logger.info(f"Pretokenizing process: {i + 1} / {num_splits}")
        else:
            for i, chunk in enumerate(chunk_generator()):
                self.merge_words(self.pretokenize(chunk, special_tokens))
                logger.info(f"Pretokenizing process: {i + 1} / {num_splits}")

        # 4. 合并
        self.build_token_to_words()
        self.merge()

    @staticmethod
    def pretokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
        str_words: dict[str, int] = {}
        safe_special_tokens = [re.escape(t) for t in special_tokens]
        mini_chunks = re.split("|".join(safe_special_tokens), chunk)
        mini_chunks = cast(list[str], mini_chunks)

        for mc in mini_chunks:
            for word_match in re.finditer(PAT, mc):
                word = word_match.group()
                str_words[word] = str_words.get(word, 0) + 1
        return {tuple(bytes([b]) for b in key.encode()): value for key, value in str_words.items()}

    def merge_words(self, words: dict[tuple[bytes, ...], int]):
        for key, value in words.items():
            self.words[key] = self.words.get(key, 0) + value

    def words_to_pairs(self) -> dict[tuple[bytes, bytes], int]:
        pairs = {}
        for word, count in self.words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def build_token_to_words(self):
        for word in self.words:
            for token in word:
                if token not in self.token_to_words:
                    self.token_to_words[token] = set()
                self.token_to_words[token].add(word)

    def merge(self):
        logger.info("Start merging...")
        pairs = self.words_to_pairs()
        heap: list[MaxHeapItem] = []
        for pair, count in pairs.items():
            heapq.heappush(heap, MaxHeapItem((pair, count)))

        while len(self.vocab) < self.vocab_size and len(pairs) >= 2:
            if len(self.vocab) % 100 == 0:
                logger.info(f"Merging: {len(self.vocab)} / {self.vocab_size}")

            # max_item = max(*pairs.items(), key=lambda x: (x[1], x[0][0], x[0][1]))
            while True:
                max_item: tuple[tuple[bytes, bytes], int] = heapq.heappop(heap).val
                if max_item[1] == pairs.get(max_item[0], 0):
                    break

            pairs.pop(max_item[0])
            self.merges.append(max_item[0])

            new_vocab = max_item[0][0] + max_item[0][1]
            if new_vocab in self.vocab_values:
                continue
            self.vocab[len(self.vocab)] = new_vocab
            self.vocab_values.add(new_vocab)
            self.token_to_words[new_vocab] = set()

            for word in self.token_to_words[max_item[0][0]].intersection(self.token_to_words[max_item[0][1]]):
                find = False
                start = 0
                replacing_idx: list[int] = []
                while start < len(word) - 1:
                    if word[start : start + 2] != max_item[0]:
                        start += 1
                        continue
                    end = start + 1

                    find = True
                    if start != 0:
                        # 在形如"ssss"的情况时，合并s+s时，由于word未更新，会导致合并后两个s时word[start-1]依旧为s
                        # 因此，如果上一个token已经被合并，此时应该用合并后的结果取代word[start-1]
                        if replacing_idx and start == replacing_idx[-1] + 2:
                            pair = (new_vocab, word[start])
                        else:
                            pair = (word[start - 1], word[start])
                        if pairs.get(pair):
                            pairs[pair] -= self.words[word]
                            heapq.heappush(heap, MaxHeapItem((pair, pairs[pair])))
                            if not pairs[pair]:
                                pairs.pop(pair)
                        pair = (pair[0], new_vocab)
                        pairs[pair] = pairs.get(pair, 0) + self.words[word]
                        heapq.heappush(heap, MaxHeapItem((pair, pairs[pair])))
                    if end != len(word) - 1:
                        pair = (word[end], word[end + 1])
                        if pairs.get(pair):
                            pairs[pair] -= self.words[word]
                            heapq.heappush(heap, MaxHeapItem((pair, pairs[pair])))
                            if not pairs[pair]:
                                pairs.pop(pair)
                        pair = (new_vocab, word[end + 1])
                        pairs[pair] = pairs.get(pair, 0) + self.words[word]
                        heapq.heappush(heap, MaxHeapItem((pair, pairs[pair])))

                    replacing_idx.append(start)
                    start += 2

                if find:
                    count = self.words.pop(word)
                    new_word: list[bytes] = []
                    start = 0
                    for idx in replacing_idx:
                        new_word += list(word[start:idx]) + [new_vocab]
                        start = idx + 2
                    new_word += list(word[start::])
                    new_word_tuple = tuple(new_word)

                    self.words[new_word_tuple] = count

                    for token in set(word):
                        if word in self.token_to_words[token]:
                            self.token_to_words[token].remove(word)
                    for token in set(new_word):
                        self.token_to_words[token].add(new_word_tuple)

    def save(self, name: str):
        encoder = gpt2_bytes_to_unicode()
        encoded_merges = []
        for merge in self.merges:
            encoded_merges.append(f"{''.join(encoder[b] for b in merge[0])} {''.join(encoder[b] for b in merge[1])}")
        with open(f"checkpoints/tokenizer/{name}_vocab.json", mode="w", encoding="utf-8") as f:
            json.dump({v.decode("latin1"): k for k, v in self.vocab.items()}, f)
        with open(f"checkpoints/tokenizer/{name}_merges.txt", mode="w", encoding="utf-8") as f:
            f.write("\n".join(encoded_merges))


if __name__ == "__main__":
    dataset = "owt"
    tokenizer = Trainer(f"data/{dataset}_train.txt", 32000, ["<|endoftext|>"], num_processes=16, num_splits=256)
    tokenizer.save(dataset)
