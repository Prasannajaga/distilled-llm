from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _extract_text(args: tuple[dict, str]) -> str:
    row, column = args
    return (row.get(column) or "").strip()


class PackedDatasetBuilder(Dataset):

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        block_size: int,
        output_path: str = "data",
        subset: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        num_workers: int = 0,
        batch_size: int = 1024,
        **load_kwargs: Any,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.output_path = Path(output_path)
        self.subset = subset
        self.split = split
        self.text_column = text_column
        self.num_workers = num_workers if num_workers > 0 else (os.cpu_count() or 1)
        self.batch_size = batch_size
        self.load_kwargs = load_kwargs

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.eos_id: int = self._resolve_eos(tokenizer)
        self.packed: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self.packed is None:
            return 0
        return len(self.packed)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.packed[idx].astype(np.int64)
        input_ids = torch.from_numpy(row[:-1].copy())
        targets = torch.from_numpy(row[1:].copy())
        return {"input_ids": input_ids, "targets": targets}

    @property
    def total_tokens(self) -> int:
        if self.packed is None:
            return 0
        return self.packed.size

    @property
    def total_sequences(self) -> int:
        return len(self)

    def build(self) -> PackedDatasetBuilder:
        raw_dataset = self._load_dataset()
        self.packed = self._tokenize_and_pack(raw_dataset)
        self._save_binary()
        return self

    def to_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = True,
        pin_memory: bool = True,
    ) -> DataLoader:
        if self.packed is None:
            raise RuntimeError("Call .build() before .to_dataloader()")
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    # ------------------------------------------------------------------ #
    #  Pipeline stages                                                    #
    # ------------------------------------------------------------------ #

    def _load_dataset(self) -> Any:
        from datasets import load_dataset

        print(f"[PACK] Loading dataset: {self.dataset_name}")
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            cache_dir=str(self.output_path),
            **self.load_kwargs,
        )
        print(f"[PACK] Loaded {len(ds):,} rows")
        return ds

    def _tokenize_and_pack(self, dataset: Any) -> np.ndarray:
        use_batch = self._supports_encode_batch(self.tokenizer)
        mode = "batch" if use_batch else "sequential"
        print(
            f"[PACK] Tokenizing: mode={mode}, "
            f"workers={self.num_workers}, batch_size={self.batch_size}"
        )

        t_start = time.monotonic()
        all_chunks: list[np.ndarray] = []
        total_tokens = 0
        n_rows = len(dataset)
        processed = 0

        for batch_start in range(0, n_rows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_rows)
            rows = dataset[batch_start:batch_end]

            texts = self._extract_texts(
                rows, self.text_column, batch_end - batch_start
            )

            valid_texts = [t for t in texts if t]
            if not valid_texts:
                processed += batch_end - batch_start
                continue

            if use_batch:
                all_ids = self._encode_batch(self.tokenizer, valid_texts)
            else:
                all_ids = [self.tokenizer.encode(t) for t in valid_texts]

            flat: list[int] = []
            for ids in all_ids:
                flat.extend(ids)
                flat.append(self.eos_id)

            all_chunks.append(np.array(flat, dtype=np.uint32))
            total_tokens += len(flat)
            processed += batch_end - batch_start

            if processed % (self.batch_size * 10) < self.batch_size:
                elapsed = time.monotonic() - t_start
                rps = processed / elapsed if elapsed > 0 else 0
                print(
                    f"[PACK] {processed:>8,}/{n_rows:,} rows | "
                    f"{total_tokens:>12,} tokens | "
                    f"{rps:>6.0f} rows/s | "
                    f"{elapsed:>6.1f}s"
                )

        token_array = (
            np.concatenate(all_chunks) if all_chunks
            else np.array([], dtype=np.uint32)
        )
        total_tokens = len(token_array)
        n_sequences = total_tokens // self.block_size
        trimmed = token_array[: n_sequences * self.block_size]
        packed = trimmed.reshape(n_sequences, self.block_size)

        elapsed = time.monotonic() - t_start
        print(
            f"[PACK] Done: {processed:,} rows -> {total_tokens:,} tokens "
            f"-> {n_sequences:,} sequences (block_size={self.block_size}) "
            f"in {elapsed:.1f}s"
        )
        return packed

    def _save_binary(self) -> None:
        bin_path = self.output_path / "data.bin"
        idx_path = self.output_path / "data.idx"

        self.packed.astype(np.uint32).tofile(str(bin_path))

        offsets = np.arange(
            0, len(self.packed) * self.block_size, self.block_size, dtype=np.uint64
        )
        offsets.tofile(str(idx_path))

        bin_mb = bin_path.stat().st_size / (1024 * 1024)
        print(f"[PACK] Saved {bin_mb:.1f} MB:")
        print(f"[PACK]   {bin_path}")
        print(f"[PACK]   {idx_path}")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _extract_texts(
        self,
        rows: dict[str, list],
        text_column: str,
        count: int,
    ) -> list[str]:
        if isinstance(rows, dict) and text_column in rows:
            raw = rows[text_column]
            if isinstance(raw, list):
                return [(t or "").strip() for t in raw]

        row_dicts = [{k: rows[k][i] for k in rows} for i in range(count)]

        if self.num_workers > 1 and count >= 64:
            args = [(r, text_column) for r in row_dicts]
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                return list(pool.map(_extract_text, args, chunksize=64))

        return [(r.get(text_column) or "").strip() for r in row_dicts]

    @staticmethod
    def _resolve_eos(tokenizer: Any) -> int:
        for attr in ("eos_id", "eos_token_id"):
            val = getattr(tokenizer, attr, None)
            if val is not None:
                return int(val)
        return 0

    @staticmethod
    def _supports_encode_batch(tokenizer: Any) -> bool:
        if hasattr(tokenizer, "encode_batch"):
            return True
        if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
            return True
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None and hasattr(inner, "encode_batch"):
            return True
        return False

    @staticmethod
    def _encode_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None and hasattr(inner, "encode_batch"):
            encodings = inner.encode_batch(texts, add_special_tokens=False)
            return [enc.ids for enc in encodings]
        if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
            batch_out = tokenizer(texts, add_special_tokens=False)
            return batch_out["input_ids"]
        if hasattr(tokenizer, "encode_batch"):
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            return [enc.ids for enc in encodings]
        return [tokenizer.encode(t) for t in texts]
