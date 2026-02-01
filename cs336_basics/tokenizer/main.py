import os
import json
from typing import BinaryIO, cast
import multiprocessing

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


class Tokenizer:
    vocab: dict[int, bytes]
    vocab_size: int
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    words: dict[tuple[bytes, ...], int]

    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        splitter: str = "<|endoftext|>",
        num_processes: int = 4,
    ):
        # 1. 词表设置与默认填充
        self.vocab_size = vocab_size
        self.vocab = {i: i.to_bytes() for i in range(256)}
        for i in range(len(special_tokens)):
            self.vocab[256 + i] = special_tokens[i].encode()

        # 2. 初始化成员变量
        self.merges = []
        self.special_tokens = special_tokens
        self.words = {}

        # 3. 并行切分文件内容，进行pretokenize
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            with open(input_path, "rb") as f:
                boundaries = find_chunk_boundaries(f, num_processes, splitter.encode())

                # The following is a serial implementation, but you can parallelize this
                # by sending each start/end pair to a set of processes.
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    logger.info(f"Sending pretokenizing file: {start} - {end} / {boundaries[-1]}")
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
                    results.append(pool.apply_async(self.pretokenize, (chunk, self.special_tokens)))
            for i in range(len(results)):
                r = results[i]
                self.merge_words(r.get())
                logger.info(f"Pretokenizing process: {i} / {num_processes}")

        # 4. 合并
        self.merge()

    @staticmethod
    def pretokenize(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
        words: dict[tuple[bytes, ...], int] = {}
        safe_special_tokens = [re.escape(t) for t in special_tokens]
        mini_chunks = re.split("|".join(safe_special_tokens), chunk)
        mini_chunks = cast(list[str], mini_chunks)

        for mc in mini_chunks:
            for word_match in re.finditer(PAT, mc):
                word = word_match.group()
                byte_tuple = tuple(bytes([b]) for b in word.encode())
                words[byte_tuple] = words.get(byte_tuple, 0) + 1
        return words

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

    def merge(self):
        logger.info("Start merging...")
        pairs = self.words_to_pairs()
        while len(self.vocab) < self.vocab_size and len(pairs) >= 2:
            if len(self.vocab) % 100 == 0:
                logger.info(f"Merging: {len(self.vocab)} / {self.vocab_size}")

            max_item = max(*pairs.items(), key=lambda x: (x[1], x[0][0], x[0][1]))
            pairs.pop(max_item[0])
            self.merges.append(max_item[0])

            new_vocab = max_item[0][0] + max_item[0][1]
            self.vocab[len(self.vocab)] = new_vocab
            for word in list(self.words.keys()):
                find = False
                start = 0
                replacing_idx: list[int] = []
                while start < len(word) - 1:
                    if word[start : start + 2] != max_item[0]:
                        start += 1
                        continue
                    end = start + 1

                    find = True
                    replacing_idx.append(start)
                    if start != 0:
                        pair = (word[start - 1], word[start])
                        if pairs.get(pair):
                            pairs[pair] -= self.words[word]
                            if not pairs[pair]:
                                pairs.pop(pair)
                        pairs[(word[start - 1], new_vocab)] = (
                            pairs.get((word[start - 1], new_vocab), 0) + self.words[word]
                        )
                    if end != len(word) - 1:
                        pair = (word[end], word[end + 1])
                        if pairs.get(pair):
                            pairs[pair] -= self.words[word]
                            if not pairs[pair]:
                                pairs.pop(pair)
                        pairs[(new_vocab, word[end + 1])] = pairs.get((new_vocab, word[end + 1]), 0) + self.words[word]

                    start += 2

                if find:
                    count = self.words.pop(word)
                    new_word: list[bytes] = []
                    start = 0
                    for idx in replacing_idx:
                        new_word += list(word[start:idx]) + [new_vocab]
                        start = idx + 2
                    new_word += list(word[start::])

                    self.words[tuple(new_word)] = count


if __name__ == "__main__":
    tokenizer = Tokenizer("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], num_processes=16)

    # save
    vocab, merges = tokenizer.vocab, tokenizer.merges
    encoder = gpt2_bytes_to_unicode()
    encoded_vocab, encoded_merges = {}, []
    for key, value in vocab.items():
        new_value = "".join([encoder[b] for b in value])
        encoded_vocab[key] = new_value
    for merge in merges:
        encoded_merges.append(("".join(encoder[b] for b in merge[0]), "".join(encoder[b] for b in merge[1])))
    with open("checkpoints/tokenizer/TinyStories.json", "w", encoding="utf-8") as f:
        json.dump({"vocab": encoded_vocab, "merges": encoded_merges}, f, ensure_ascii=False, indent=4)
