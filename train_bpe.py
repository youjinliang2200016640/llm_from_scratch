from cs336_basics.tokenization import train_bpe_tokenizer
from cs336_basics.utils import timer
import pickle

if __name__ == "__main__":
    # Train a BPE tokenizer on the given text file
    with timer("Training BPE tokenizer") as t_info:
        vocab, merges = train_bpe_tokenizer(
            input_path="data/TinyStoriesV2-GPT4-valid.txt",
            vocab_size=10_000,
            special_tokens=["<|endoftext|>"],
        )

    # Save the trained tokenizer to a directory
    with open("assets/bpe_tokenizer.pkl", "wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "training_time": t_info,
            },
            f,
        )