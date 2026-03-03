import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transformer import (
    GroupedQueryAttention,
    KVCache,
    RMSNorm,
    RotaryPositionEmbedding,
    SWIGLU_FFN,
)

# @title GQA_transformer

class GQARopeAttention(GroupedQueryAttention):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, block_size: int, dropout: float = 0.0):
        super().__init__(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, block_size=block_size, dropout=dropout)
        self.head_dim = n_embd // n_head
        self.rope = RotaryPositionEmbedding(self.head_dim, block_size)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout_p = float(dropout)
        self.qk_norm_eps = 1e-6
        self.q_scale = nn.Parameter(
            torch.full((self.n_head, 1, 1), self.head_dim ** 0.5)
        )
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")
        sdpa_doc = getattr(F.scaled_dot_product_attention, "__doc__", "") if self.has_sdpa else ""
        self.sdpa_has_gqa = "enable_gqa" in (sdpa_doc or "")

    def _causal_mask(self, t_q: int, t_k: int, position_offset: int, device: torch.device) -> torch.Tensor:
        if position_offset == 0 and t_q == t_k:
            return self.tril[:t_q, :t_k].to(device=device, dtype=torch.bool)
        q_pos = position_offset + torch.arange(t_q, device=device).unsqueeze(1)
        k_pos = torch.arange(t_k, device=device).unsqueeze(0)
        return k_pos <= q_pos

    def forward(self, x, kv_cache=None, position_offset=0):
        B, T, C = x.shape

        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k = self.k_proj(x)
        v = self.v_proj(x)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, position_offset=position_offset)
        q = F.normalize(q, p=2.0, dim=-1, eps=self.qk_norm_eps)
        k = F.normalize(k, p=2.0, dim=-1, eps=self.qk_norm_eps)
        q = q * self.q_scale

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.has_sdpa:
            sdpa_kwargs = {
                "attn_mask": None,
                "dropout_p": self.attn_dropout_p if self.training else 0.0,
                "is_causal": True,
            }
            use_native_gqa = self.groups > 1 and self.sdpa_has_gqa

            if kv_cache is not None:
                t_q = q.size(-2)
                t_k = k.size(-2)
                if position_offset > 0 or t_q != t_k:
                    sdpa_kwargs["attn_mask"] = self._causal_mask(t_q, t_k, position_offset, q.device)
                    sdpa_kwargs["is_causal"] = False

            if use_native_gqa:
                sdpa_kwargs["enable_gqa"] = True
            elif self.groups > 1:
                k = k.repeat_interleave(self.groups, dim=1)
                v = v.repeat_interleave(self.groups, dim=1)

            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                **sdpa_kwargs,
            )
        else:
            if self.groups > 1:
                k = k.repeat_interleave(self.groups, dim=1)
                v = v.repeat_interleave(self.groups, dim=1)

            att = (q @ k.transpose(-2, -1)) * self.scale
            Tq = q.size(-2)
            Tk = k.size(-2)
            mask = self._causal_mask(Tq, Tk, position_offset, att.device)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            attn = att @ v

        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class GQABlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, block_size, dropout=0.0):
        super().__init__()
        self.attn = GQARopeAttention(n_embd, n_head, n_kv_head, block_size, dropout)
        self.ffn = SWIGLU_FFN(n_embd, expansion_ratio=2.7, dropout=dropout)
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        position_offset = kv_cache.seq_len if kv_cache is not None else 0
        x = x + self.dropout(self.attn(self.norm1(x), kv_cache=kv_cache, position_offset=position_offset))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GQATransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        n_emb,
        n_head,
        n_kv_head,
        vocab_size,
        block_size,
        dropout=0.0,
    ):
        super().__init__()

        assert n_emb % n_head == 0
        assert n_head % n_kv_head == 0

        self.n_kv_head = n_kv_head
        self.head_dim = n_emb // n_head
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                GQABlock(n_emb, n_head, n_kv_head, block_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def build_kv_cache(self, max_batch_size, device=None, dtype=None):
        if device is None:
            device = self.token_emb.weight.device
        if dtype is None:
            dtype = self.token_emb.weight.dtype
        return [
            KVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.block_size,
                n_kv_heads=self.n_kv_head,
                head_dim=self.head_dim,
                dtype=dtype,
            ).to(device)
            for _ in range(len(self.blocks))
        ]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        x = self.drop(self.token_emb(idx))

        if kv_cache is not None:
            assert len(kv_cache) == len(self.blocks)

        for i, block in enumerate(self.blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            x = block(x, kv_cache=cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                label_smoothing=0.0,
            )

        return logits, loss

    def resize_token_embeddings(self, new_vocab_size):
        old_vocab_size, n_emb = self.token_emb.weight.shape
        if new_vocab_size == old_vocab_size:
            return

        device = self.token_emb.weight.device
        dtype = self.token_emb.weight.dtype

        new_token_emb = nn.Embedding(new_vocab_size, n_emb, device=device, dtype=dtype)
        torch.nn.init.normal_(new_token_emb.weight, mean=0.0, std=0.02)

        num_to_copy = min(old_vocab_size, new_vocab_size)
        with torch.no_grad():
            new_token_emb.weight[:num_to_copy].copy_(self.token_emb.weight[:num_to_copy])

        self.token_emb = new_token_emb
        new_lm_head = nn.Linear(n_emb, new_vocab_size, bias=False, device=device, dtype=dtype)
        torch.nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_lm_head.weight[:num_to_copy].copy_(self.lm_head.weight[:num_to_copy])
        self.lm_head = new_lm_head
        self.lm_head.weight = self.token_emb.weight

    def _sample_next_token(self, logits, idx, temperature, top_k, repetition_penalty):
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if repetition_penalty != 1.0:
            penalized = logits.gather(1, idx) / repetition_penalty
            logits = logits.scatter(1, idx, penalized)

        if top_k is not None:
            k = min(int(top_k), logits.size(-1))
            if k > 0:
                threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.7,
        top_k=None,
        repetition_penalty=1.2,
        eos_token_id=None,
        use_cache=True,
    ):
        self.eval()
        idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]

        if use_cache:
            max_cache_new_tokens = self.block_size - idx_cond.size(1)
            if max_new_tokens > max_cache_new_tokens:
                use_cache = False

        kv_cache = None
        logits = None
        if use_cache:
            kv_cache = self.build_kv_cache(
                max_batch_size=idx_cond.size(0),
                device=idx_cond.device,
                dtype=self.token_emb.weight.dtype,
            )
            logits, _ = self.forward(idx_cond, kv_cache=kv_cache)

        for _ in range(max_new_tokens):
            if not use_cache:
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
                logits, _ = self.forward(idx_cond)

            next_token_logits = logits[:, -1, :]
            idx_next = self._sample_next_token(
                next_token_logits,
                idx,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            if eos_token_id is not None and torch.all(idx_next.squeeze(-1).eq(eos_token_id)):
                break

            idx = torch.cat([idx, idx_next], dim=1)
            if use_cache:
                logits, _ = self.forward(idx_next, kv_cache=kv_cache)

        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
