import torch
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from collections.abc import Callable
import math

class F:
    @staticmethod
    def softmax(x: Float[torch.Tensor, " ..."], dim: int) -> torch.Tensor:
        """Compute the softmax of the input tensor along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): Dimension along which to compute the softmax.

        Returns:
            torch.Tensor: Tensor with the same shape as input, with softmax applied along the specified dimension.
        """
        exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        return exp_x / sum_exp_x
    
    @staticmethod
    def logsoftmax(x: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute the log-softmax of the input tensor along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): Dimension along which to compute the log-softmax.
        Returns:
            torch.Tensor: Tensor with the same shape as input, with log-softmax applied along the specified dimension.
        """
        max_x = torch.max(x, dim=dim, keepdim=True).values
        log_sum_exp_x = torch.log(torch.sum(torch.exp(x - max_x), dim=dim, keepdim=True))
        return x - max_x - log_sum_exp_x
    
    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        """Compute the SiLU (Sigmoid Linear Unit) activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with the same shape as input, with SiLU applied element-wise.
        """
        return x * torch.sigmoid(x)
    
    @staticmethod
    def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... values d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
    ) -> Float[torch.Tensor, " ... queries d_v"]:
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        d_k = Q.size(-1)
        scores:torch.Tensor = einsum(
            Q, K, "... i d, ... j d -> ... i j"
        ) / (d_k ** 0.5)  # (..., seq_len_q, seq_len_k)
        if mask is not None:
            scores = scores.masked_fill(mask == False, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  # (..., seq_len_q, seq_len_k)
        output = einsum(
            attn_weights, V, "... i j, ... j d -> ... i d"
        ) # (..., seq_len_q, d_v)
        return output
    
    @staticmethod
    def cross_entropy_loss(
        logits: Float[torch.Tensor, " ... num_classes"],
        target: Int[torch.Tensor, " ..."],
    ) -> Float[torch.Tensor, ""]:
        """
        Given logits and target tensor, return the average cross entropy loss.

        Args:
            logits (Float[Tensor, " ... num_classes"]): Logits tensor
            target (Int[Tensor, " ..."]): Target tensor
        Returns:
            Float[Tensor, ""]: Average cross entropy loss
        """
        log_probs = F.logsoftmax(logits, dim=-1)  # (..., num_classes)
        target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # (...)
        loss = -target_log_probs.mean()  # Scalar
        return loss

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=std, 
            a=-3*std, 
            b=3*std
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = Wx
            W (d_out, d_in)
        Args:
            x (torch.Tensor): (batch, d_in)

        Returns:
            torch.Tensor: (batch, d_out)
        """
        return einsum(
            self.weight, x, "o i, ... i -> ... o"
        )
        
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=1.0, 
            a=-3.0, 
            b=3.0
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """y = W[token_ids]
            W (num_embeddings, embedding_dim)
        Args:
            token_ids (torch.Tensor): (...,)

        Returns:
            torch.Tensor: (..., embedding_dim)
        """
        return self.weight[token_ids]
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            d_model (int): final dimension of the input
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-5.
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        ) # [d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape

        Args:
            x (torch.Tensor): [batch_size, sequence_length, d_model]

        Returns:
            torch.Tensor: [batch_size, sequence_length, d_model]
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        ) # [batch_size, sequence_length, 1]
        x_normed = x / rms # [batch_size, sequence_length, d_model]
        x_normed = x_normed.to(in_type)
        return einsum(
            x_normed, 
            self.weight,
            "... d, d -> ... d"
        )
        
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class Softmax(nn.Module):
    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimension along which to compute the softmax.
        """
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        """
        Args:
            d_model (int): Dimension of the input and output
            d_ffn (int): Dimension of the hidden layer in the feedforward network
        """
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.w1 = Linear(d_model, d_ffn)
        self.w2 = Linear(d_ffn, d_model)
        self.w3 = Linear(d_model, d_ffn)
        self.activation = SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """W2(SiLU(W1(x)) * W3(x))

        Args:
            x (torch.Tensor): [batch_size, sequence_length, d_model]

        Returns:
            torch.Tensor: [batch_size, sequence_length, d_model]
        """
        x1 = self.w1(x)  # [batch_size, sequence_length, d_ffn]
        x3 = self.w3(x)  # [batch_size, sequence_length, d_ffn]
        return self.w2(
            einsum(
                self.activation(x1), 
                x3, 
                "... d, ... d -> ... d"
            )
        )

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos_seq = torch.arange(0, max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i , j -> i j", pos_seq, inv_freq)
        self.register_buffer("cos_cached", torch.cos(sinusoid_inp), persistent=False)
        self.register_buffer("sin_cached", torch.sin(sinusoid_inp), persistent=False)
    
    def forward(self, x: Float[torch.Tensor, " ... sequence_length d_k"], token_positions: Int[torch.Tensor, " ... sequence_length"]) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
            Note that you should tolerate x with an arbitrary number of batch dimensions. You should
            assume that the token positions are a tensor of shape (..., seq_len) specifying the token
            positions of x along the sequence dimension

        Args:
            x (torch.Tensor): _description_
            token_positions (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2) # type: ignore
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2) # type: ignore
        x1 = x[..., ::2]  # (..., seq_len, d_k/2)
        x2 = x[..., 1::2]  # (..., seq_len, d_k/2)
        return torch.stack(
            [
                x1 * cos - x2 * sin, 
                x1 * sin + x2 * cos
            ], 
            dim=-1
        ).flatten(-2)  # (..., seq_len, d_k)
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None, rope: RotaryPositionalEmbedding | None = None):
        """
        Args:
            d_model (int): Dimension of the input and output
            num_heads (int): Number of attention heads
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
            
        
    def forward(self, x: torch.Tensor, token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape

        Args:
            x (torch.Tensor): [batch_size, sequence_length, d_model]

        Returns:
            torch.Tensor: [batch_size, sequence_length, d_model]
        """
        B, S, _ = x.shape
        Q = rearrange(
            self.q_proj(x), 
            "b s (h d) -> b h s d", 
            h=self.num_heads
        )
        K = rearrange(
            self.k_proj(x), 
            "b s (h d) -> b h s d", 
            h=self.num_heads
        )
        V = rearrange(
            self.v_proj(x), 
            "b s (h d) -> b h s d", 
            h=self.num_heads
        )       
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)  # [B, S]
            Q = self.rope(Q, token_positions)  # [B, num_heads, S, d_k]
            K = self.rope(K, token_positions)  # [B, num_heads, S, d_k]     
        mask = torch.tril(
            torch.ones((S, S), device=x.device, dtype=torch.bool), 
            diagonal=0
        )  # [S, S]
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
        attn_output = F.scaled_dot_product_attention(Q, K, V, mask)  # [B, num_heads, S, d_k]
        attn_output = rearrange(
            attn_output, 
            "b h s d -> b s (h d)", 
            h=self.num_heads
        )  # [B, S, d_model]
        output = self.output_proj(attn_output)  # [B, S, d_model]
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None, rope: RotaryPositionalEmbedding | None = None):
        """
        Args:
            d_model (int): Dimension of the input and output
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of the hidden layer in the feedforward network
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype, rope=rope)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape

        Args:
            x (torch.Tensor): [batch_size, sequence_length, d_model]

        Returns:
            torch.Tensor: [batch_size, sequence_length, d_model]
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, context_length: int, rope_theta: float = 10000.0,  device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the input and output
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of the hidden layer in the feedforward network
            num_layers (int): Number of transformer blocks
            max_seq_len (int): Maximum sequence length
            device (torch.device | None, optional): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, rope=self.rope) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length) and return logits of shape (batch_size, sequence_length, vocab_size)

        Args:
            token_ids (torch.Tensor): [batch_size, sequence_length]

        Returns:
            torch.Tensor: [batch_size, sequence_length, vocab_size]
        """
        x = self.token_embeddings(token_ids)  # [B, S, d_model]
        for layer in self.layers:
            x = layer(x)  # [B, S, d_model]
        x = self.ln_final(x)  # [B, S, d_model]
        logits = self.lm_head(x)  # [B, S, vocab_size]
        return logits
    


        