import os
from typing import BinaryIO, cast

import regex as re

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


def pairs_greater(x: tuple[bytes, bytes, int], y: tuple[bytes, bytes, int]):
    if x[2] != y[2]:
        return x[2] > y[2]

    if x[0] != y[0]:
        return x[0] > y[0]

    return x[1] >= y[1]


class Tokenizer:
    vocab: dict[int, bytes]
    vocab_size: int
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    words: dict[tuple[bytes, ...], int]

    def __init__(
        self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], splitter: str = "<|endoftext|>"
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

        # 3. 切分文件内容，进行pretokenize
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, splitter.encode())

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                self.pretokenize(chunk)

        # 4. 合并
        self.merge()

    def pretokenize(self, chunk: str):
        safe_special_tokens = [re.escape(t) for t in self.special_tokens]
        mini_chunks = re.split("|".join(safe_special_tokens), chunk)
        mini_chunks = cast(list[str], mini_chunks)

        for mc in mini_chunks:
            for word_match in re.finditer(PAT, mc):
                word = word_match.group()
                byte_tuple = tuple(bytes([b]) for b in word.encode())
                self.words[byte_tuple] = self.words.get(byte_tuple, 0) + 1

    def words_to_pairs(self) -> dict[tuple[bytes, bytes], int]:
        pairs = {}
        for word, count in self.words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def merge(self):
        pairs = self.words_to_pairs()
        while len(self.vocab) < self.vocab_size and len(pairs) >= 2:
            max_item = max(*pairs.items(), key=lambda x: (x[1], x[0][0], x[0][1]))
            pairs.pop(max_item[0])
            self.merges.append(max_item[0])

            new_vocab = max_item[0][0] + max_item[0][1]
            self.vocab[len(self.vocab)] = new_vocab
            for word in list(self.words.keys()):
                find = False
                new_word: list[bytes] = []
                start = 0
                while start < len(word):
                    end = start + 1
                    if word[start : start + 2] != max_item[0] or start == len(word) - 1:
                        new_word.append(word[start])
                        start += 1
                        continue

                    find = True
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

                    new_word.append(new_vocab)
                    start += 2

                if find:
                    count = self.words.pop(word)
                    self.words[tuple(new_word)] = count


if __name__ == "__main__":
    # tokenizer = Tokenizer("data/TinyStoriesV2-GPT4-valid.txt", 30000, ["<|endoftext|>"])
    tokenizer = Tokenizer("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    print(tokenizer.words)
