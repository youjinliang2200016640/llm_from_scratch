import torch
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np
import os
import typing
import time
from contextlib import contextmanager

@contextmanager
def timer(name=""):
    info = {}
    start = time.perf_counter_ns()
    yield info
    end = time.perf_counter_ns()
    elapse = end - start
    info["start"] = start
    info["end"] = end
    info["elapse"] = elapse
    

def clip_grads_with_norm_(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip gradients of the given parameters in-place to have a maximum L2 norm of `max_l2_norm`.

    Args:
        parameters (Iterable[torch.nn.Parameter]): An iterable of model parameters.
        max_l2_norm (float): The maximum allowed L2 norm for the gradients.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm < max_l2_norm:
        return
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
                
class DataLoader:
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device



def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    inputs = torch.LongTensor(batch_size, context_length)
    labels = torch.LongTensor(batch_size, context_length)
    dataset_length = len(dataset)
    for i in range(batch_size):
        start_idx = np.random.randint(0, dataset_length - context_length)
        end_idx = start_idx + context_length
        inputs[i] = torch.from_numpy(dataset[start_idx:end_idx])
        labels[i] = torch.from_numpy(dataset[start_idx + 1:end_idx + 1])
    return inputs.to(device), labels.to(device)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """Saves a model and optimizer checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current training iteration.
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The output file path or file-like object.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Loads a model and optimizer checkpoint.

    Args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The source file path or file-like object.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        Tuple containing the model, optimizer, and the iteration number from the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration