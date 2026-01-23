"""
BPE (Byte Pair Encoding) Tokenizer 实现
"""

import os
import regex as re
from collections import defaultdict, Counter
from cs336_basics.pretokenization import PAT, pretokenization


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE tokenizer"""
    special_tokens = special_tokens or []
    
    # Step 1: 读取文件并预分词
    total_count = pretokenization(
        input_path,
        num_processes=num_processes,
        special_tokens=special_tokens,
    )
    
    # Step 2: 转换为bytes列表形式
    words: list[list[bytes]] = [] # 每个word是bytes的列表
    word_freqs: list[int] = [] # 每个word的频率
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set) # pair到包含该pair的word_ids映射
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int) # pair的总频率
    
    for word, count in total_count.items():
        word_bytes = [bytes([b]) for b in word.encode("utf-8")]
        word_id = len(words)
        words.append(word_bytes)
        word_freqs.append(count)
        
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            pair_to_words[pair].add(word_id)
            pair_counts[pair] += count
    
    # Step 3: 初始化词表
    vocab: dict[int, bytes] = {}
    token_id = 0    
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    for st in special_tokens:
        vocab[token_id] = st.encode("utf-8")
        token_id += 1    
    
    # Step 4: 迭代合并
    merges: list[tuple[bytes, bytes]] = []
    
    while token_id < vocab_size and pair_counts:
        # 找最高频的pair (使用字典序作为tie-breaker)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]
        
        if best_count <= 0:
            break
        
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[token_id] = new_token
        token_id += 1
        
        # 获取包含这个pair的所有word_ids
        affected_word_ids = pair_to_words.pop(best_pair, set())
        del pair_counts[best_pair]
        
        for word_id in affected_word_ids:
            word = words[word_id]
            freq = word_freqs[word_id]
            
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    # 移除受影响的pairs
                    if i > 0:
                        left_pair = (word[i - 1], word[i])
                        if left_pair in pair_counts:
                            pair_counts[left_pair] -= freq
                            if pair_counts[left_pair] <= 0:
                                del pair_counts[left_pair]
                                pair_to_words.pop(left_pair, None)
                    
                    if i + 2 < len(word):
                        right_pair = (word[i + 1], word[i + 2])
                        if right_pair in pair_counts:
                            pair_counts[right_pair] -= freq
                            if pair_counts[right_pair] <= 0:
                                del pair_counts[right_pair]
                                pair_to_words.pop(right_pair, None)
                    
                    # 执行合并
                    word[i] = new_token
                    del word[i + 1]
                    
                    # 添加新的pairs
                    if i > 0:
                        new_left_pair = (word[i - 1], new_token)
                        pair_counts[new_left_pair] += freq
                        pair_to_words[new_left_pair].add(word_id)
                    
                    if i + 1 < len(word):
                        new_right_pair = (new_token, word[i + 1])
                        pair_counts[new_right_pair] += freq
                        pair_to_words[new_right_pair].add(word_id)
                else:
                    i += 1
    
    return vocab, merges


class BPETokenizer:
    """BPE Tokenizer 实现"""
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.token_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_priority: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        self._build_special_token_pattern()
        self.pretokenize_pattern = re.compile(PAT, re.UNICODE)
    
    def _build_special_token_pattern(self):
        if not self.special_tokens:
            self.special_token_pattern = None
            return
        
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_tokens]
        pattern = "|".join(escaped)
        self.special_token_pattern = re.compile(f"({pattern})")
    
    def _split_by_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        if not self.special_token_pattern:
            return [(text, False)]
        
        result = []
        last_end = 0
        
        for match in self.special_token_pattern.finditer(text):
            start, end = match.span()
            if start > last_end:
                result.append((text[last_end:start], False))
            result.append((match.group(), True))
            last_end = end
        
        if last_end < len(text):
            result.append((text[last_end:], False))
        
        return result
    
    def _apply_bpe(self, token_bytes: tuple[bytes, ...]) -> list[bytes]:
        if len(token_bytes) <= 1:
            return list(token_bytes)
        
        word = list(token_bytes)
        
        while len(word) > 1:
            min_priority = float("inf")
            min_idx = -1
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < min_priority:
                        min_priority = priority
                        min_idx = i
            
            if min_idx == -1:
                break
            
            new_token = word[min_idx] + word[min_idx + 1]
            word = word[:min_idx] + [new_token] + word[min_idx + 2:]
        
        return word
    
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        ids = []
        segments = self._split_by_special_tokens(text)
        
        for segment, is_special in segments:
            if not segment:
                continue
                
            if is_special:
                token_bytes = segment.encode("utf-8")
                if token_bytes in self.token_to_id:
                    ids.append(self.token_to_id[token_bytes])
            else:
                for match in self.pretokenize_pattern.finditer(segment):
                    pretoken = match.group()
                    token_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    merged = self._apply_bpe(token_bytes)
                    for token in merged:
                        if token in self.token_to_id:
                            ids.append(self.token_to_id[token])
        
        return ids
    
    def decode(self, ids: list[int]) -> str:
        byte_list = []
        for id in ids:
            if id in self.vocab:
                byte_list.append(self.vocab[id])
        
        all_bytes = b"".join(byte_list)
        return all_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable) :
        for text in iterable:
            for id in self.encode(text):
                yield id
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """从文件加载BPE Tokenizer"""
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "rb") as vf:
            for line in vf:
                line = line.rstrip(b"\n")
                token_id_str, token_bytes = line.split(b"\t")
                token_id = int(token_id_str)
                vocab[token_id] = token_bytes
        
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as mf:
            for line in mf:
                line = line.rstrip(b"\n")
                if not line or line.startswith(b"#"):
                    continue
                part1, part2 = line.split(b" ")
                merges.append((part1, part2))
        
        return cls(vocab, merges, special_tokens)