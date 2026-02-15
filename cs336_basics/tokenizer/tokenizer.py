import json
import heapq
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.utils import gpt2_bytes_to_unicode


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
INF = 999999999


class HeapItem:
    def __init__(self, val: bytes):
        self.val = val
        self.idx = INF

        self.pre: HeapItem | None = None
        self.next: HeapItem | None = None
        self.valid = True

    def __lt__(self, other: "HeapItem"):
        return self.idx < other.idx

    def __eq__(self, other: "HeapItem"):
        return self.idx == other.idx


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.word_cache: dict[str, list[int]] = {}
        self.bytes_to_merges: dict[bytes, set[tuple[bytes, bytes]]] = {}
        self.merge_to_id = {merge: i for i, merge in enumerate(self.merges)}

        if special_tokens:
            values = self.vocab.values()
            for sp in special_tokens:
                if sp in values:
                    vocab[len(vocab)] = sp.encode()
        self.rev_vocab = {v: k for k, v in vocab.items()}

        for merge in merges:
            for token in merge:
                if token not in self.bytes_to_merges:
                    self.bytes_to_merges[token] = set()
                self.bytes_to_merges[token].add(merge)

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
        vocab = {v: k.encode("latin1") for k, v in vocab_data.items()}
        with open(merges_filepath, encoding="utf-8") as f:
            merges_data = f.readlines()
        merges: list[tuple[bytes, bytes]] = []
        for line in merges_data:
            x, y = line.split()
            merges.append((bytes([decoder[token] for token in x]), bytes([decoder[token] for token in y])))
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
        for word in re.findall(PAT, chunk):
            word_bytes = word.encode()
            if word not in self.word_cache:
                if word_bytes in self.rev_vocab:
                    self.word_cache[word] = [self.rev_vocab[word_bytes]]
                else:
                    word_tuple = tuple(word_bytes[i : i + 1] for i in range(len(word_bytes)))
                    self.word_cache[word] = self._encode_word(word_tuple)
            result += self.word_cache[word]
        return result

    def _encode_word(self, word: tuple[bytes, ...]) -> list[int]:
        word_list = list(word)
        heap = [HeapItem(i) for i in word_list]
        for i in range(len(heap) - 1):
            heap[i].next = heap[i + 1]
            if (word_list[i], word_list[i + 1]) in self.merge_to_id:
                heap[i].idx = self.merge_to_id[(word_list[i], word_list[i + 1])]
        for i in range(1, len(heap)):
            heap[i].pre = heap[i - 1]
        head = heap[0]
        heapq.heapify(heap)
        while True:
            max_item = heapq.heappop(heap)
            if not max_item.next:
                break
            if not max_item.valid or max_item.idx == INF:
                continue

            nxt = max_item.next
            nxt.valid = False
            max_item.val += nxt.val
            max_item.next = nxt.next
            max_item.idx = INF
            if max_item.next:
                max_item.next.pre = max_item
                if (max_item.val, max_item.next.val) in self.merge_to_id:
                    max_item.idx = self.merge_to_id[(max_item.val, max_item.next.val)]
            if max_item.pre:
                if (max_item.pre.val, max_item.val) in self.merge_to_id:
                    max_item.pre.idx = self.merge_to_id[(max_item.pre.val, max_item.val)]
                else:
                    max_item.pre.idx = INF
            heapq.heappush(heap, max_item)

        result = []
        while head:
            result.append(self.rev_vocab[head.val])
            head = head.next
        return result
        # word_list = list(word)
        # while len(word_list) > 1:
        #     min_id = 10 * len(self.merges)
        #     min_pos = -1

        #     for i in range(len(word_list) - 1):
        #         pair = (word_list[i], word_list[i + 1])
        #         if pair in self.merge_to_id:
        #             id = self.merge_to_id[pair]
        #             if id < min_id:
        #                 min_pos, min_id = i, id

        #     if min_id == 10 * len(self.merges):
        #         break

        #     word_list[min_pos] = word_list[min_pos] + word_list[min_pos + 1]
        #     del word_list[min_pos + 1]
        # return [self.rev_vocab[token] for token in word_list]


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
