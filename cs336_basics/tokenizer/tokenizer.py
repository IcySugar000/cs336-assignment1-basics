import json
from typing import cast
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.utils import gpt2_bytes_to_unicode


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.word_cache: dict[str, list[int]] = {}

        if special_tokens:
            values = self.vocab.values()
            for sp in special_tokens:
                if sp in values:
                    vocab[len(vocab)] = sp.encode()
        self.rev_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        decoder = {v: k.to_bytes() for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
        vocab = {v: k.encode() for k, v in vocab_data.items()}
        with open(merges_filepath) as f:
            merges_data = f.readlines()
        merges: list[tuple[bytes, bytes]] = []
        for line in merges_data:
            x, y = line.split()
            xb, yb = b"", b""
            for c in x:
                xb += decoder[c]
            for c in y:
                yb += decoder[c]
            merges.append((xb, yb))
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_chunk(text)

        result: list[int] = []
        special_tokens = sorted(
            self.special_tokens, key=lambda x: len(x), reverse=True
        )  # 避免overlap token出问题，需要长的token放置在前
        pattern = "|".join(re.escape(t) for t in special_tokens)
        start = 0
        for match in re.finditer(pattern, text):
            chunk = text[start : match.start()]
            if chunk:
                result += self._encode_chunk(chunk)
            start = match.end()

            sp = match.group()
            result.append(self.rev_vocab[sp.encode()])
        result += self._encode_chunk(text[start:])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            ids = self.encode(chunk)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode(errors="replace")

    def _encode_chunk(self, chunk: str) -> list[int]:
        """No special token in chunk"""
        result: list[int] = []
        for match in re.finditer(PAT, chunk):
            word = match.group()
            word_bytes = tuple(bytes([b]) for b in word.encode())
            if word_bytes not in self.word_cache:
                self.word_cache[word] = self._encode_word(word_bytes)
            result += self.word_cache[word]
        return result

    def _encode_word(self, word: tuple[bytes, ...]) -> list[int]:
        for merge in self.merges:
            find = False
            replacing_idx: list[int] = []
            for i in range(len(word) - 1):
                if merge == (word[i], word[i + 1]):
                    find = True
                    replacing_idx.append(i)
            if find:
                new_word: list[bytes] = []
                start = 0
                for idx in replacing_idx:
                    new_word += list(word[start:idx])
                    new_word.append(merge[0] + merge[1])
                    start = idx + 2
                new_word += list(word[start:])
                word = tuple(new_word)

        return [self.rev_vocab[token] for token in word]


if __name__ == "__main__":
    tokenizer = Tokenizer(
        vocab={
            0: b" ",
            1: b"a",
            2: b"c",
            3: b"e",
            4: b"h",
            5: b"t",
            6: b"th",
            7: b" c",
            8: b" a",
            9: b"the",
            10: b" at",
        },
        merges=[(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")],
    )
    print(tokenizer.encode("the cat ate"))
    print(tokenizer.decode([9, 7, 1, 5, 10, 3]))
