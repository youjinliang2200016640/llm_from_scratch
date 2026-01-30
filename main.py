from argparse import ArgumentParser
import torch
from cs336_basics.module import TransformerLM
from cs336_basics.optim import AdamWOptimizer
from cs336_basics.tokenization import BPETokenizer

def main(args):
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--validate_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    main(args)