
import torch
import torch.nn as nn
import torch.nn.functional as F

# @title GroupedQueryAttention
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    """

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, block_size: int = 512, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.groups = n_head // n_kv_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)

        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # ---- Q ----
        q = self.q_proj(x)                       # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim)
        q = q.transpose(1, 2)                    # (B, Hq, T, D)

        # ---- K, V ----
        k = self.k_proj(x)                       # (B, T, Hkv*D)
        v = self.v_proj(x)

        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        k = k.transpose(1, 2)                    # (B, Hkv, T, D)
        v = v.transpose(1, 2)                    # (B, Hkv, T, D)

        # ---- Expand K/V to match Q heads ----
        k = k.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, D)
        v = v.repeat_interleave(self.groups, dim=1)  # (B, Hq, T, D)

        # ---- Attention ----
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, Hq, T, T)

        att = att.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # ---- Output ----
        out = att @ v                            # (B, Hq, T, D)
        out = out.transpose(1, 2).contiguous()   # (B, T, Hq, D)
        out = out.view(B, T, C)                  # (B, T, C)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out

# @title RopeEmbeddings
class RotaryPositionEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        total_len = position_offset + seq_len

        if total_len > self.cos_cached.shape[0]:
            self._build_cache(total_len)

        cos = self.cos_cached[position_offset:total_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[position_offset:total_len].unsqueeze(0).unsqueeze(0)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

# @title KVcache
class KVCache(nn.Module):

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.current_len = 0

        cache_shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = k_new.shape[0]
        new_len = k_new.shape[2]
        end_pos = self.current_len + new_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: tried to write up to position {end_pos}, "
                f"but max_seq_len is {self.max_seq_len}"
            )

        self.k_cache[:batch_size, :, self.current_len:end_pos, :] = k_new
        self.v_cache[:batch_size, :, self.current_len:end_pos, :] = v_new
        self.current_len = end_pos

        return (
            self.k_cache[:batch_size, :, :end_pos, :],
            self.v_cache[:batch_size, :, :end_pos, :],
        )

    @property
    def seq_len(self) -> int:
        return self.current_len

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0

    def resize(self, new_max_seq_len: int):
        if new_max_seq_len == self.max_seq_len:
            return

        old_len = min(self.current_len, new_max_seq_len)
        batch_size, n_kv_heads, _, head_dim = self.k_cache.shape

        new_k = torch.zeros(
            batch_size, n_kv_heads, new_max_seq_len, head_dim,
            dtype=self.k_cache.dtype, device=self.k_cache.device,
        )
        new_v = torch.zeros(
            batch_size, n_kv_heads, new_max_seq_len, head_dim,
            dtype=self.v_cache.dtype, device=self.v_cache.device,
        )

        if old_len > 0:
            new_k[:, :, :old_len, :] = self.k_cache[:, :, :old_len, :]
            new_v[:, :, :old_len, :] = self.v_cache[:, :, :old_len, :]

        self.k_cache = new_k
        self.v_cache = new_v
        self.max_seq_len = new_max_seq_len
        self.current_len = old_len

#@title Activation function
class SWIGLU_FFN(nn.Module):
    """SwiGLU Feed-Forward Network (used in LLaMA-style models)"""

    def __init__(self, n_embd: int, expansion_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(expansion_ratio * n_embd)

        self.w1 = nn.Linear(n_embd, hidden_dim)
        self.w2 = nn.Linear(n_embd, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.w3(F.silu(self.w1(x)) * self.w2(x))
        )

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight