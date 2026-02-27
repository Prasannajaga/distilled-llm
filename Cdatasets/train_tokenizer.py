"""Train a custom BPE tokenizer on the OpenWebMath dataset.

Uses HuggingFace `tokenizers` (BpeTrainer) with math-aware pre-tokenization
and the project's special tokens. All configurable values come from
`scripts..config.TokenizerConfig`.

Usage:
    python -m scripts..train_tokenizer [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from utils.config import TokenizerConfig, default_tokenizer_config
from utils.constants import (
    MATH_LATEX_TOKENS,
    SPECIAL_TOKENS,
    TOKENIZER_ARTIFACT_FILENAME,
    TOKENIZER_CONFIG_FILENAME,
)


def _build_base_tokenizer() -> Tokenizer:
    """Construct a blank BPE tokenizer with math-aware pre-tokenization."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace("\r\n", "\n"),
        normalizers.Replace("\r", "\n"),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])

    return tokenizer


def _build_trainer(config: TokenizerConfig) -> trainers.BpeTrainer:
    """Build a BpeTrainer with the project's special tokens and LaTeX additions."""
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        show_progress=config.show_progress,
        continuing_subword_prefix="##",
    )

    return trainer


def _stream_texts_from_dataset(config: TokenizerConfig) -> Iterator[str]:
    """Yield text strings from OpenWebMath, loading only specified parquet shards.

    When `config.data_files` is set, only those specific shard files are
    downloaded from the Hub — avoiding the full ~27 GB dataset download.
    """
    from Cdatasets import load_dataset

    print(f"[TOKENIZER] Loading dataset: {config.dataset_name}")
    if config.data_files:
        print(f"[TOKENIZER] Shard files: {config.data_files}")

    load_kwargs: dict = {
        "split": config.dataset_split,
    }
    if config.data_files:
        load_kwargs["data_files"] = list(config.data_files)
        load_kwargs["verification_mode"] = "no_checks"

    dataset = load_dataset(config.dataset_name, **load_kwargs)

    total = min(len(dataset), config.max_training_samples)  # type: ignore[arg-type]
    print(f"[TOKENIZER] Training on {total:,} samples (capped from {len(dataset):,})")

    yielded = 0
    for i in range(0, total, config.training_chunk_size):
        chunk_end = min(i + config.training_chunk_size, total)
        chunk = dataset[i:chunk_end]  # type: ignore[index]

        texts: list[str] = chunk[config.text_column]
        for text in texts:
            if text and text.strip():
                yield text
                yielded += 1

    print(f"[TOKENIZER] Yielded {yielded:,} non-empty texts for training")


def _apply_post_processor(tokenizer: Tokenizer) -> None:
    """Configure BOS/EOS template processing after training."""
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    if bos_id is None or eos_id is None:
        raise ValueError(
            "Special tokens <|bos|> or <|eos|> not found in trained vocabulary. "
            "This indicates a training failure — aborting."
        )

    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> <|bos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],  
    )


def _save_tokenizer_config(config: TokenizerConfig, output_dir: Path) -> None:
    """Persist the training configuration alongside the tokenizer artifact."""
    config_data = {
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "text_column": config.text_column,
        "training_chunk_size": config.training_chunk_size,
        "max_training_samples": config.max_training_samples,
        "special_tokens": SPECIAL_TOKENS,
        "math_latex_tokens": MATH_LATEX_TOKENS,
    }
    config_path = output_dir / TOKENIZER_CONFIG_FILENAME
    config_path.write_text(json.dumps(config_data, indent=2))
    print(f"[TOKENIZER] Config saved to {config_path}")


def train_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """Train a BPE tokenizer on the configured dataset and save artifacts.

    Returns the trained `tokenizers.Tokenizer` instance.

    Raises:
        ValueError: If critical special tokens are missing after training.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / TOKENIZER_ARTIFACT_FILENAME

    print("[TOKENIZER] ============================")
    print("[TOKENIZER]  BPE Tokenizer Training")
    print("[TOKENIZER] ============================")
    print(f"[TOKENIZER] Vocab size target : {config.vocab_size:,}")
    print(f"[TOKENIZER] Min frequency     : {config.min_frequency}")
    print(f"[TOKENIZER] Output            : {artifact_path}")
    print()

    tokenizer = _build_base_tokenizer()
    trainer = _build_trainer(config)

    text_iterator = _stream_texts_from_dataset(config)

    start_time = time.monotonic()
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    elapsed = time.monotonic() - start_time

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\n[TOKENIZER] Training complete in {elapsed:.1f}s")
    print(f"[TOKENIZER] Final vocab size: {actual_vocab_size:,}")

    _apply_post_processor(tokenizer)

    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<|pad|>"),
        pad_token="<|pad|>",
    )

    tokenizer.save(str(artifact_path))
    print(f"[TOKENIZER] Tokenizer saved to {artifact_path}")

    _save_tokenizer_config(config, output_dir)

    _print_validation_summary(tokenizer)

    return tokenizer


def _print_validation_summary(tokenizer: Tokenizer) -> None:
    """Quick sanity-check: encode a few math-heavy strings and display results."""
    test_strings = [
        "The derivative of x^2 is 2x.",
        "\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}",
        "If f(x) = \\sqrt{x^2 + 1}, find f'(x).",
        "3.14159265",
        "<|question|>What is 2+2?<|reasoning|>2+2=4<|answer|>4<|eos|>",
    ]

    print("\n[TOKENIZER] ============================")
    print("[TOKENIZER]  Validation Samples")
    print("[TOKENIZER] ============================")
    for text in test_strings:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        print(f"  Input  : {text!r}")
        print(f"  Tokens : {len(encoded.ids)} ids")
        print(f"  Decoded: {decoded!r}")
        print()


def _parse_args(argv: list[str] | None = None) -> TokenizerConfig:
    """Parse CLI arguments and return a validated TokenizerConfig."""
    defaults = default_tokenizer_config()

    parser = argparse.ArgumentParser(
        description="Train a custom BPE tokenizer on OpenWebMath",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vocab_size", type=int, default=defaults.vocab_size,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--min_frequency", type=int, default=defaults.min_frequency,
        help="Minimum token frequency to keep",
    )
    parser.add_argument(
        "--dataset_name", type=str, default=defaults.dataset_name,
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--dataset_split", type=str, default=defaults.dataset_split,
        help="Dataset split for training (supports slicing like 'train[:50000]')",
    )
    parser.add_argument(
        "--text_column", type=str, default=defaults.text_column,
        help="Name of the text column in the dataset",
    )
    parser.add_argument(
        "--training_chunk_size", type=int, default=defaults.training_chunk_size,
        help="Number of samples per internal loading chunk",
    )
    parser.add_argument(
        "--max_training_samples", type=int, default=defaults.max_training_samples,
        help="Maximum number of samples to train on",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(defaults.output_dir),
        help="Directory to save tokenizer artifacts",
    )
    parser.add_argument(
        "--no_progress", action="store_true", default=False,
        help="Disable progress bars during training",
    )
    parser.add_argument(
        "--data_files", type=str, nargs="*",
        default=list(defaults.data_files) if defaults.data_files else None,
        help="Specific parquet shard file patterns to load (e.g. 'data/train-00000-of-00114-*.parquet')",
    )

    args = parser.parse_args(argv)

    data_files_tuple: tuple[str, ...] | None = (
        tuple(args.data_files) if args.data_files else None
    )

    config = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        training_chunk_size=args.training_chunk_size,
        max_training_samples=args.max_training_samples,
        output_dir=Path(args.output_dir),
        show_progress=not args.no_progress,
        data_files=data_files_tuple,
    )

    if config.vocab_size < len(SPECIAL_TOKENS):
        print(
            f"[ERROR] vocab_size ({config.vocab_size}) must be >= "
            f"number of special tokens ({len(SPECIAL_TOKENS)})",
            file=sys.stderr,
        )
        sys.exit(1)

    return config


def main(argv: list[str] | None = None) -> None:
    config = _parse_args(argv)
    train_tokenizer(config)


if __name__ == "__main__":
    main()
