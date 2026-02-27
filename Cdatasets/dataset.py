from __future__ import annotations

import os
import random
from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from datasets import DownloadConfig, load_dataset, load_from_disk

from utils.constants import (
    AUTOMATHTEXT_COL_ABSTRACT,
    AUTOMATHTEXT_COL_TEXT,
    AUTOMATHTEXT_COL_TITLE,
)
from Cdatasets.tokenizer import MathTokenizer


class PretrainDataset(Dataset):

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: Any,
        block_size: int,
        cache_size: int = 4096,
        subset: Optional[str] = None,
        split: str = "train",
        hf_cache_dir: Optional[str | Path] = None,
        local_files_only: bool = False,
        max_stream_rows: int = 10000,
        prefer_streaming: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_dir = Path(data_dir)
        self.cache_size = max(0, int(cache_size))
        self.subset = subset
        self.split = split
        self.hf_cache_dir = str(hf_cache_dir) if hf_cache_dir is not None else None
        self.local_files_only = local_files_only
        self.max_stream_rows = int(max_stream_rows)
        self.prefer_streaming = bool(prefer_streaming)
        self._stream_tokens: Optional[list[tuple[int, ...]]] = None
        self._bos_id = self._resolve_token_id("bos_id", "bos_token_id", 0)
        self._eos_id = self._resolve_token_id("eos_id", "eos_token_id", self._bos_id)

        # Two modes:
        # 1) local saved dataset path -> load_from_disk
        # 2) HF dataset id (e.g. "math-ai/AutoMathText") -> load_dataset and reuse HF cache
        if self.data_dir.exists():
            self._hf_dataset = load_from_disk(str(self.data_dir))
        else:
            download_config = DownloadConfig(local_files_only=local_files_only)
            if self.prefer_streaming:
                self._hf_dataset = None
                self._stream_tokens = []
                stream_ds = load_dataset(
                    str(data_dir),
                    name=subset,
                    split=split,
                    cache_dir=self.hf_cache_dir,
                    streaming=True,
                    download_config=download_config,
                )
                for row in islice(stream_ds, self.max_stream_rows):
                    text = self._format_row(row)
                    tokens = self.tokenizer.encode(text) if text else []
                    if len(tokens) < 2:
                        tokens = [self._bos_id, self._eos_id]
                    # Keep only one training window per streamed row to bound RAM.
                    if len(tokens) > self.block_size + 1:
                        start = random.randrange(len(tokens) - self.block_size)
                        tokens = tokens[start : start + self.block_size + 1]
                    self._stream_tokens.append(tuple(tokens))
                if len(self._stream_tokens) == 0:
                    self._stream_tokens.append((self._bos_id, self._eos_id))
                self._size = len(self._stream_tokens)
                self._token_cache = OrderedDict()
                return

            ds = load_dataset(
                str(data_dir),
                name=subset,
                split=split,
                cache_dir=self.hf_cache_dir,
                download_mode="reuse_dataset_if_exists",
                download_config=download_config,
            )
            self._hf_dataset = ds

        self._size = len(self._hf_dataset)

        self._token_cache: OrderedDict[int, tuple[int, ...]] = OrderedDict()

    def _resolve_token_id(self, custom_attr: str, hf_attr: str, default: int) -> int:
        value = getattr(self.tokenizer, custom_attr, None)
        if value is None:
            value = getattr(self.tokenizer, hf_attr, None)
        return int(value) if value is not None else int(default)

    def _get_or_encode_tokens(self, idx: int) -> list[int]:
        if self._stream_tokens is not None:
            return list(self._stream_tokens[idx])

        if idx in self._token_cache:
            cached = self._token_cache.pop(idx)
            self._token_cache[idx] = cached
            return list(cached)

        row = self._hf_dataset[idx]
        text = self._format_row(row)
        tokens = self.tokenizer.encode(text) if text else []
        if len(tokens) < 2:
            tokens = [self._bos_id, self._eos_id]

        frozen = tuple(tokens)
        if self.cache_size > 0:
            self._token_cache[idx] = frozen
            if len(self._token_cache) > self.cache_size:
                self._token_cache.popitem(last=False)
        return list(frozen)

    @staticmethod
    def _format_row(row: dict) -> str:
        """Combine available AutoMathText fields into a single passage.

        Schema: text, meta, title, url, abstract.
        We use title + abstract as optional preamble, followed by the
        main text body.
        """
        parts: list[str] = []

        title = (row.get(AUTOMATHTEXT_COL_TITLE) or "").strip()
        if title:
            parts.append(title)

        abstract = (row.get(AUTOMATHTEXT_COL_ABSTRACT) or "").strip()
        if abstract:
            parts.append(abstract)

        body = (row.get(AUTOMATHTEXT_COL_TEXT) or "").strip()
        if body:
            parts.append(body)

        return "\n\n".join(parts)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self._get_or_encode_tokens(idx)

        if len(tokens) > self.block_size + 1:
            start = random.randrange(len(tokens) - self.block_size)
            tokens = tokens[start : start + self.block_size + 1]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "targets": targets}


class FineTuneDataset(Dataset):

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: MathTokenizer,
        block_size: int,
        subset: Optional[str] = None,
        split: str = "train",
        hf_cache_dir: Optional[str | Path] = None,
        local_files_only: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_dir = Path(data_dir) 
        if self.data_dir.exists():
            raw_dataset = load_from_disk(str(self.data_dir))
        else:
            download_config = DownloadConfig(local_files_only=local_files_only)
            raw_dataset = load_dataset(
                str(data_dir),
                name=subset,
                split=split,
                cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
                download_mode="reuse_dataset_if_exists",
                download_config=download_config,
            )
        self.samples: list[list[int]] = []

        for row in raw_dataset:
            formatted = self._format_row(row)
            token_ids = tokenizer.encode(formatted)
            if len(token_ids) >= 2:
                self.samples.append(token_ids)

    def _format_row(self, row: dict) -> str:
        problem = row.get("problem", row.get("question", ""))
        solution = row.get("solution", row.get("chain_of_thought", ""))
        answer = row.get("answer", row.get("final_answer", ""))

        if not answer and solution:
            lines = solution.strip().split("\n")
            answer = lines[-1] if lines else ""

        return (
            f"<|question|>{problem}"
            f"<|reasoning|>{solution}"
            f"<|answer|>{answer}<|eos|>"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[idx]
        if len(tokens) > self.block_size + 1:
            tokens = tokens[: self.block_size + 1]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "targets": targets}


def pretrain_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_list = [item["input_ids"] for item in batch]
    target_list = [item["targets"] for item in batch]

    max_len = max(x.size(0) for x in input_ids_list)

    padded_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (inp, tgt) in enumerate(zip(input_ids_list, target_list)):
        seq_len = inp.size(0)
        padded_inputs[i, :seq_len] = inp
        padded_targets[i, :seq_len] = tgt
        attention_mask[i, :seq_len] = True

    return padded_inputs, padded_targets, attention_mask


def finetune_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return pretrain_collate_fn(batch, pad_id=pad_id)


def download_dataset(
    name: str,
    output_dir: str | Path,
    split: str = "train",
    subset: Optional[str] = None,
    skip: bool = False
) -> Path:
    """Download a HuggingFace dataset and save it to disk.

    Uses streaming mode to avoid shard-tracking bugs in datasets >=4.x
    when loading multi-directory JSONL configs (e.g. AutoMathText
    ``arxiv-0.70-to-1.00``).
    """
    output_dir = Path(output_dir)
    if not skip and output_dir.exists() and any(output_dir.iterdir()):
        print(f"[SKIP] Dataset already exists at {output_dir}")
        return output_dir

    from datasets import Dataset as HFDataset
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"split": split, "streaming": True}
    if subset is not None:
        kwargs["name"] = subset

    print(f"[DOWNLOAD] Loading {name} (split={split}, streaming)...")
    iterable_ds = load_dataset(name, **kwargs)

    ds = HFDataset.from_generator(
        lambda: (row for row in iterable_ds),
    )

    ds.save_to_disk(str(output_dir))
    print(f"[DONE] Saved {len(ds)} examples to {output_dir}")
    return output_dir
