import os
import json
import pickle
from typing import BinaryIO
import regex as re
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

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

def get_pretokenized_text_counts(text : str) -> Counter[str]:
    """
    Pre-tokenize the input text and return a count of each pre-token.
    """
    pattern = re.compile(PAT, flags=re.UNICODE)
    count = Counter()
    for token in pattern.finditer(text):
        count[token.group(0)] += 1
    return count

def pretokenization(
    input_path : str | os.PathLike,
    special_tokens: list[str],
    num_processes : int = 4
)-> Counter[str]:
    with open(input_path, 'rb') as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        num_workers = min(cpu_count(), num_processes)
        special_tokens = [re.escape(token) for token in special_tokens]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                text = f.read(end - start).decode('utf-8', errors='ignore')
                if special_tokens:
                    chunks = re.split("|".join(special_tokens), text, flags=re.UNICODE)
                else:
                    chunks = [text]
                for chunk in chunks:
                    futures.append(executor.submit(get_pretokenized_text_counts, chunk))
            total_count = Counter()
            for future in futures:
                total_count.update(future.result())
    return total_count


def train_bpe_tokenizer(
    input_path : str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_process: int = 4,
):
    special_tokens = special_tokens or []
    
    # Step 1: 读取文件并预分词
    total_count = pretokenization(
        input_path,
        special_tokens=special_tokens,
        num_processes=num_process,
    )
    
    # Step 2: 转换为bytes列表形式
    words: list[list[bytes]] = [] # 每个word是bytes列表
    word_freqs: list[int] = [] # 每个word的频率
    pair_in_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set) # pair到包含该pair的word_ids映射
    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int) # pair的总频率
        
    for word, count in total_count.items():
        word_bytes = [bytes([b]) for b in word.encode('utf-8')]
        word_id = len(words)
        words.append(word_bytes)
        word_freqs.append(count)
        
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i+1])
            pair_counts[pair] += count
            pair_in_words[pair].add(word_id)
    
    # Step 3: 初始化词表
    vocab: dict[int, bytes] = {}
    token_id: int = 0
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    for st in special_tokens:
        vocab[token_id] = st.encode('utf-8')
        token_id += 1
    
    # Step 4: 迭代合并
    merges: list[tuple[bytes, bytes]] = []
    while token_id < vocab_size and pair_counts:
        # 找最高频的pair (使用字典序作为tie-breaker)
        most_pair = max(pair_counts, key = lambda p:(pair_counts[p], p))
        most_count = pair_counts[most_pair]
        
        if most_count <= 0:
            break
        
        new_token = most_pair[0] + most_pair[1]
        merges.append(most_pair)
        vocab[token_id] = new_token
        token_id += 1
        
        # 获取包含这个pair的所有word_ids
        affected_word_ids = pair_in_words.pop(most_pair, set())
        del pair_counts[most_pair]
        
        for word_id in affected_word_ids:
            word = words[word_id]
            freq = word_freqs[word_id]
            
            i = 0
            while i < len(word) - 1:
                if word[i] == most_pair[0] and word[i + 1] == most_pair[1]:
                    # 移除受影响的pairs
                    if i > 0:
                        left_pair = (word[i - 1], word[i])
                        if left_pair in pair_counts:
                            pair_counts[left_pair] -= freq
                            if pair_counts[left_pair] <= 0:
                                del pair_counts[left_pair]
                                pair_in_words.pop(left_pair, None)
                    if i + 2 < len(word):
                        right_pair = (word[i + 1], word[i + 2])
                        if right_pair in pair_counts:
                            pair_counts[right_pair] -= freq
                            if pair_counts[right_pair] <= 0:
                                del pair_counts[right_pair]
                                pair_in_words.pop(right_pair, None)
                    
                    # 执行合并
                    word[i] = new_token
                    del word[i + 1]
                    
                    # 添加新的pairs
                    if i > 0:
                        new_left_pair = (word[i - 1], new_token)
                        pair_counts[new_left_pair] += freq
                        pair_in_words[new_left_pair].add(word_id)
                        
                    if i + 1 < len(word):
                        new_right_pair = (new_token, word[i + 1])
                        pair_counts[new_right_pair] += freq
                        pair_in_words[new_right_pair].add(word_id)
                        
                else:
                    i += 1
    return vocab, merges
 
class BPETokenizer:
    """
    BPE Tokenizer 实现
    
    支持内存高效的流式编码：
    - encode: 标准编码方法
    - encode_iterable: 流式编码，内存复杂度为O(1)
    
    流式编码的关键是在预分词边界处分割文本，确保token不会跨越chunk边界。
    对于大文件，我们逐块处理，并在块之间保留可能不完整的预分词。
    """
    
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
            word[min_idx] = new_token
            del word[min_idx + 1]
        
        return word
    
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """对单个预分词进行BPE编码"""
        token_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        merged = self._apply_bpe(token_bytes)
        ids = []
        for token in merged:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
        return ids
    
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
                    ids.extend(self._encode_pretoken(pretoken))
        
        return ids
    
    def decode(self, ids: list[int]) -> str:
        byte_list = []
        for id in ids:
            if id in self.vocab:
                byte_list.append(self.vocab[id])
        
        all_bytes = b"".join(byte_list)
        return all_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable):
        """
        内存高效的流式编码
        
        对于大文件，使用迭代器逐块处理，确保内存复杂度为O(1)。
        关键是在预分词边界处分割，确保token不会跨越chunk边界。
        
        算法:
        1. 对每个chunk，与上一个chunk末尾的carry_over合并
        2. 按特殊token分割文本
        3. 对普通文本进行预分词匹配
        4. 除了最后一个预分词外，其他都直接编码
        5. 最后一个预分词可能不完整，保留到下一个chunk
        
        Args:
            iterable: 可迭代对象，每个元素是一个文本块（字符串）
            
        Yields:
            token ID
        """
        carry_over = ""
        
        for chunk in iterable:
            # 合并carry_over和当前chunk
            text = carry_over + chunk
            carry_over = ""
            
            if not text:
                continue
            
            # 按特殊token分割
            segments = self._split_by_special_tokens(text)
            
            # 处理每个segment
            for i, (segment, is_special) in enumerate(segments):
                if not segment:
                    continue
                
                is_last_segment = (i == len(segments) - 1)
                
                if is_special:
                    token_bytes = segment.encode("utf-8")
                    if token_bytes in self.token_to_id:
                        yield self.token_to_id[token_bytes]
                else:
                    # 预分词
                    matches = list(self.pretokenize_pattern.finditer(segment))
                    
                    if not matches:
                        if is_last_segment:
                            # 没有匹配到预分词，可能是不完整的，保留
                            carry_over = segment
                        continue
                    
                    # 检查segment末尾是否被完全匹配
                    last_match = matches[-1]
                    fully_matched = (last_match.end() == len(segment))
                    
                    for j, match in enumerate(matches):
                        is_last_match = (j == len(matches) - 1)
                        pretoken = match.group()
                        
                        if is_last_segment and is_last_match and fully_matched:
                            # 最后一个segment的最后一个预分词，且匹配到了末尾
                            # 这个预分词可能不完整，保留到下一个chunk
                            carry_over = pretoken
                        else:
                            # 编码当前预分词
                            for token_id in self._encode_pretoken(pretoken):
                                yield token_id
                    
                    # 如果segment末尾有未匹配的文本，保留它
                    if is_last_segment and not fully_matched:
                        carry_over = segment[last_match.end():]
        
        # 处理最后的carry_over
        if carry_over:
            # 按特殊token分割
            segments = self._split_by_special_tokens(carry_over)
            for segment, is_special in segments:
                if not segment:
                    continue
                if is_special:
                    token_bytes = segment.encode("utf-8")
                    if token_bytes in self.token_to_id:
                        yield self.token_to_id[token_bytes]
                else:
                    for match in self.pretokenize_pattern.finditer(segment):
                        pretoken = match.group()
                        for token_id in self._encode_pretoken(pretoken):
                            yield token_id
    
    @classmethod
    def from_files(cls, path: str | os.PathLike):
        """从文件加载BPE Tokenizer"""
        path = Path(path)
        if path.suffix == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            vocab = {int(k): bytes.fromhex(v) for k, v in data['vocab'].items()}
            merges = [(bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in data['merges']]
            special_tokens = data.get('special_tokens', [])
            return cls(vocab, merges, special_tokens)
            
    
    def save_to_files(self, path: str | os.PathLike):
        """将BPE Tokenizer保存到文件"""
        pass
            
                
        

