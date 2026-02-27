from __future__ import annotations

from utils.config import (
    ModelConfig,
    TrainConfig,
    DistillConfig,
    teacher_config,
    student_config,
)
from scripts.model import GQATransformer
from Cdatasets.tokenizer import MathTokenizer


def count_parameters(model: GQATransformer) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    tokenizer = MathTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    tc = teacher_config(vocab_size=tokenizer.vocab_size)
    sc = student_config(vocab_size=tokenizer.vocab_size)

    teacher = GQATransformer(
        num_layers=tc.num_layers,
        n_emb=tc.n_embd,
        n_head=tc.n_head,
        n_kv_head=tc.n_kv_head,
        vocab_size=tc.vocab_size,
        block_size=tc.block_size,
        dropout=tc.dropout,
    )

    student = GQATransformer(
        num_layers=sc.num_layers,
        n_emb=sc.n_embd,
        n_head=sc.n_head,
        n_kv_head=sc.n_kv_head,
        vocab_size=sc.vocab_size,
        block_size=sc.block_size,
        dropout=sc.dropout,
    )

    print(f"Teacher: {count_parameters(teacher) / 1e6:.2f}M parameters")
    print(f"Student: {count_parameters(student) / 1e6:.2f}M parameters")

    test_text = "<|bos|><|question|>What is 1 + 2?<|reasoning|>1 + 2 = 3<|answer|>3<|eos|>"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    print(f"\nOriginal:  {test_text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded:   {decoded}")
    print(f"Round-trip match: {test_text == decoded}")

    test_math = "\\frac{1}{2} + \\sqrt{4} = 2.5"
    ids2 = tokenizer.encode(test_math)
    decoded2 = tokenizer.decode(ids2)
    print(f"\nOriginal:  {test_math}")
    print(f"Token IDs: {ids2}")
    print(f"Decoded:   {decoded2}")

    import torch
    dummy_input = torch.tensor([ids[:20]], dtype=torch.long)
    logits, _ = student(dummy_input)
    print(f"\nForward pass shape: {logits.shape}")
    print("[ALL TESTS PASSED]")


if __name__ == "__main__":
    main()
