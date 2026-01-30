import math
from collections.abc import Callable
import torch

class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        """
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float): Learning rate
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
    def step(self, closure: Callable | None = None): # type: ignore
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                d_p = p.grad.data
                p.data = p.data - lr / math.sqrt(t + 1) * d_p
                state['t'] = t + 1
        return loss
    
class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        """
        Args:
            params (iterable): Iterable of parameters to optimize
            lr (float): Learning rate
            betas (tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.01.
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure: Callable | None = None): # type: ignore
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['first_moment'] = torch.zeros_like(p.data)
                    state['second_moment'] = torch.zeros_like(p.data)
                first_moment:torch.Tensor = state['first_moment']
                second_moment:torch.Tensor = state['second_moment']
                state['step'] += 1
                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                adapted_lr = lr * (bias_correction2 ** 0.5) / bias_correction1
                denom = second_moment.sqrt().add_(eps)
                step_size = adapted_lr / denom
                p.data = p.data - step_size * first_moment
                if weight_decay != 0:
                    p.data = p.data - lr * weight_decay * p.data
        return loss


class LrScheduler:
    @staticmethod
    def cosine_annealing( 
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ) -> float:
        """Cosine annealing learning rate scheduler with warmup.

        Args:
            it (int): Current iteration.
            max_learning_rate (float): Maximum learning rate.
            min_learning_rate (float): Minimum learning rate.
            warmup_iters (int): Number of warmup iterations.
            cosine_cycle_iters (int): Number of iterations for one cosine cycle.

        Returns:
            float: Adjusted learning rate.
        """
        if it < warmup_iters:
            lr = max_learning_rate * it / warmup_iters
        elif warmup_iters <= it <= cosine_cycle_iters:
            lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate) 
        else:
            lr = min_learning_rate
        return lr