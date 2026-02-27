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

    def forward(self, x, kv_cache=None, position_offset=0):
        B, T, C = x.shape

        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k = self.k_proj(x)
        v = self.v_proj(x)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, position_offset=position_offset)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            Tq = q.size(-2)
            Tk = k.size(-2)
            if position_offset == 0 and Tq == Tk:
                mask = torch.tril(torch.ones(Tq, Tk, device=att.device, dtype=torch.bool))
            else:
                q_pos = position_offset + torch.arange(Tq, device=att.device).unsqueeze(1)
                k_pos = torch.arange(Tk, device=att.device).unsqueeze(0)
                mask = k_pos <= q_pos
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
        B, T = idx.shape
        assert T <= self.block_size
        assert idx.max() < self.token_emb.num_embeddings

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
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.0)

        return logits, loss

    def resize_token_embeddings(self, new_vocab_size):
        old_vocab_size, n_emb = self.token_emb.weight.shape
        if new_vocab_size == old_vocab_size:
            return

        new_token_emb = nn.Embedding(new_vocab_size, n_emb)
        torch.nn.init.normal_(new_token_emb.weight, mean=0.0, std=0.02)

        num_to_copy = min(old_vocab_size, new_vocab_size)
        new_token_emb.weight.data[:num_to_copy] = self.token_emb.weight.data[:num_to_copy]

        self.token_emb = new_token_emb
        new_lm_head = nn.Linear(n_emb, new_vocab_size, bias=False)
        new_lm_head.weight.data[:num_to_copy] = self.lm_head.weight.data[:num_to_copy]
        self.lm_head = new_lm_head
        self.lm_head.weight = self.token_emb.weight

    @torch.no_grad()
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

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
                logits, _ = self.forward(idx_cond)
                logits = logits[:, -1, :]

                if temperature == 0.0:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if repetition_penalty != 1.0:
                        unique_tokens = torch.unique(idx)
                        logits[:, unique_tokens] = logits[:, unique_tokens] / repetition_penalty

                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                if eos_token_id is not None and idx_next.item() == eos_token_id:
                    break

                idx = torch.cat([idx, idx_next], dim=1)

            return idx

        idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
        kv_cache = self.build_kv_cache(
            max_batch_size=idx_cond.size(0),
            device=idx_cond.device,
            dtype=self.token_emb.weight.dtype,
        )
        logits, _ = self.forward(idx_cond, kv_cache=kv_cache)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if repetition_penalty != 1.0:
                    unique_tokens = torch.unique(idx)
                    logits[:, unique_tokens] = logits[:, unique_tokens] / repetition_penalty

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

            idx = torch.cat([idx, idx_next], dim=1)
            logits, _ = self.forward(idx_next, kv_cache=kv_cache)

        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
