import os
from typing import BinaryIO
import regex as re
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

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

def get_pretokenize_text_counter(text: str) -> Counter[str]:
    """
    Pre-tokenize the input text and return a count of each pre-token.
    """
    pattern = re.compile(PAT, re.UNICODE)
    count = Counter()
    for match in pattern.finditer(text):
        token = match.group(0)
        count[token] += 1
    return count
        
    


def pretokenization(file_path: str | os.PathLike, num_processes: int, special_tokens: list[str],):
    ## Usage
    with open(file_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        special_tokens = [re.escape(token) for token in special_tokens]
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                text = f.read(end - start).decode("utf-8", errors="ignore")
                # strip out all special tokens from your corpus
                chunks = re.split("|".join(special_tokens), text, flags=re.UNICODE, concurrent=True)
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                for chunk in chunks:
                    assert '<|' not in chunk, f"{start} to {end} contains special token \n{chunk}"
                    futures.append(executor.submit(get_pretokenize_text_counter, chunk))
            total_count = Counter()
            for future in futures:
                total_count.update(future.result())
    return total_count


            

        
    
    
