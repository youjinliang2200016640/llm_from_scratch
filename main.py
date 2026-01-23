from cs336_basics.pretokenization import pretokenization

if __name__ == "__main__":
    total_counter =  pretokenization(
    "tests/fixtures/tinystories_sample_5M.txt",
    4,
    ["<|endoftext|>"]
)
    print(total_counter.most_common(20))