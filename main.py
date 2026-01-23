from cs336_basics.tokenization import pretokenization

if __name__ == "__main__":
    total_counter =  pretokenization(
    "tests/fixtures/tinystories_sample_5M.txt",
    ["<|endoftext|>"],
    4
)
    print(total_counter.most_common(20))