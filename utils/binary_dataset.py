from __future__ import annotations

import hashlib
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.constants import (
    AUTOMATHTEXT_COL_ABSTRACT,
    AUTOMATHTEXT_COL_TEXT,
    AUTOMATHTEXT_COL_TITLE,
)

HEADER_MAGIC = b"TBIN"
HEADER_VERSION = 1
HEADER_SIZE = 16

_BATCH_SIZE = 1024
_FLUSH_THRESHOLD = 500_000
_LOG_INTERVAL = 5_000


def _format_row_standalone(row: dict) -> str:
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


class TokenizedBinaryDataset(Dataset):
    """Tokenize-once, load-from-binary dataset wrapper.

    First call tokenizes every row, concatenates into a single 1-D token
    stream, and writes it as a uint16 or uint32 .bin file.  Subsequent
    calls memory-map the existing .bin, giving zero-copy random access
    with negligible RAM overhead.

    The binary layout is::

        [4 bytes] magic b"TBIN"
        [4 bytes] version (uint32 LE)
        [4 bytes] block_size (uint32 LE)
        [4 bytes] dtype code: 2 = uint16, 4 = uint32
        [N * dtype_bytes] flat token stream

    Parameters
    ----------
    source : str | Path
        HuggingFace dataset identifier **or** path to a saved Arrow dataset
        on disk.  When the path exists locally it is loaded with
        ``load_from_disk``; otherwise ``load_dataset`` is used.
    tokenizer : Any
        Tokenizer exposing an ``encode(text) -> list[int]`` method.
    block_size : int
        Context window length.  Each sample yields ``block_size`` input
        tokens and ``block_size`` target tokens (shifted by one).
    bin_path : str | Path
        Destination for the binary file.  If it already exists **and**
        has a valid header whose ``block_size`` matches, tokenization is
        skipped entirely.
    subset : str | None
        Optional dataset config / subset name passed to ``load_dataset``.
    split : str
        Dataset split (default ``"train"``).
    max_rows : int | None
        Cap the number of rows read from the source (useful for debugging
        or streaming large datasets).  ``None`` means no limit.
    num_workers : int
        Number of parallel workers for text formatting. 0 means use all
        available CPU cores.
    """

    def __init__(
        self,
        source: str | Path,
        tokenizer: Any,
        block_size: int,
        bin_path: str | Path,
        subset: Optional[str] = None,
        split: str = "train",
        max_rows: Optional[int] = None,
        deduplicate: bool = False,
        num_workers: int = 0,
    ) -> None:
        self.block_size = block_size
        self.bin_path = Path(bin_path)

        if self._is_valid_cache(self.bin_path, block_size):
            print(f"[BIN] Loading pre-tokenized binary: {self.bin_path}")
        else:
            self._tokenize_and_save(
                source=source,
                tokenizer=tokenizer,
                block_size=block_size,
                subset=subset,
                split=split,
                max_rows=max_rows,
                deduplicate=deduplicate,
                num_workers=num_workers,
            )

        self._mmap, self._dtype = self._load_mmap(self.bin_path)
        self._n_tokens = len(self._mmap)
        self._n_samples = max(0, (self._n_tokens - 1) // self.block_size)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start_idx = idx * self.block_size
        chunk = self._mmap[start_idx : start_idx + self.block_size + 1].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        targets = torch.from_numpy(chunk[1:].copy())
        return {"input_ids": input_ids, "targets": targets}

    @staticmethod
    def _format_row(row: dict) -> str:
        return _format_row_standalone(row)

    @staticmethod
    def _choose_dtype(max_id: int) -> np.dtype:
        if max_id < np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    @staticmethod
    def _is_valid_cache(path: Path, block_size: int) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != HEADER_MAGIC:
                    return False
                version = struct.unpack("<I", f.read(4))[0]
                if version != HEADER_VERSION:
                    return False
                stored_bs = struct.unpack("<I", f.read(4))[0]
                if stored_bs != block_size:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _load_mmap(path: Path) -> tuple[np.ndarray, np.dtype]:
        with open(path, "rb") as f:
            f.read(4)
            f.read(4)
            f.read(4)
            dtype_code = struct.unpack("<I", f.read(4))[0]
        dt = np.dtype(np.uint16) if dtype_code == 2 else np.dtype(np.uint32)
        return np.memmap(path, dtype=dt, mode="r", offset=HEADER_SIZE), dt

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

    def _tokenize_and_save(
        self,
        source: str | Path,
        tokenizer: Any,
        block_size: int,
        subset: Optional[str],
        split: str,
        max_rows: Optional[int],
        deduplicate: bool = False,
        num_workers: int = 0,
    ) -> None:
        source_path = Path(source)

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(tokenizer)
        dt = self._choose_dtype(vocab_size)

        bos_id = self._resolve_token_id(tokenizer, "bos_id", "bos_token_id", 0)
        eos_id = self._resolve_token_id(tokenizer, "eos_id", "eos_token_id", bos_id)

        self.bin_path.parent.mkdir(parents=True, exist_ok=True)

        use_batch = self._supports_encode_batch(tokenizer)
        n_workers = num_workers if num_workers > 0 else os.cpu_count() or 1
        n_format_workers = max(1, n_workers)

        buffer: list[np.ndarray] = []
        buffer_len = 0
        total_tokens = 0
        row_count = 0
        skipped = 0
        duplicates_skipped = 0
        t_start = time.monotonic()

        seen_hashes: set[int] | None = set() if deduplicate else None

        mode = "batch" if use_batch else "sequential"
        print(f"[BIN] Starting tokenization -> {self.bin_path}")
        print(
            f"[BIN]   dtype={dt}, block_size={block_size}, "
            f"mode={mode}, format_workers={n_format_workers}, "
            f"batch_size={_BATCH_SIZE}"
        )

        with open(self.bin_path, "wb") as f:
            f.write(HEADER_MAGIC)
            f.write(struct.pack("<I", HEADER_VERSION))
            f.write(struct.pack("<I", block_size))
            f.write(struct.pack("<I", dt.itemsize))

            row_batch: list[dict] = []

            for row in self._load_rows(source_path, subset, split, max_rows):
                row_batch.append(row)

                if len(row_batch) < _BATCH_SIZE:
                    continue

                batch_tokens, batch_skipped, batch_dup = self._process_batch(
                    row_batch, tokenizer, bos_id, eos_id, dt,
                    use_batch, seen_hashes, n_format_workers,
                )
                row_count += len(row_batch) - batch_skipped - batch_dup
                skipped += batch_skipped
                duplicates_skipped += batch_dup
                row_batch.clear()

                if batch_tokens is not None:
                    buffer.append(batch_tokens)
                    buffer_len += len(batch_tokens)

                if buffer_len >= _FLUSH_THRESHOLD:
                    merged = np.concatenate(buffer)
                    f.write(merged.tobytes())
                    total_tokens += len(merged)
                    buffer.clear()
                    buffer_len = 0
                    f.flush()

                if row_count > 0 and row_count % _LOG_INTERVAL < _BATCH_SIZE:
                    self._log_progress(
                        row_count, total_tokens, buffer_len, dt, t_start
                    )

            if row_batch:
                batch_tokens, batch_skipped, batch_dup = self._process_batch(
                    row_batch, tokenizer, bos_id, eos_id, dt,
                    use_batch, seen_hashes, n_format_workers,
                )
                row_count += len(row_batch) - batch_skipped - batch_dup
                skipped += batch_skipped
                duplicates_skipped += batch_dup
                row_batch.clear()

                if batch_tokens is not None:
                    buffer.append(batch_tokens)
                    buffer_len += len(batch_tokens)

            if buffer:
                merged = np.concatenate(buffer)
                f.write(merged.tobytes())
                total_tokens += len(merged)
                buffer.clear()
                buffer_len = 0
                f.flush()

        if total_tokens == 0:
            with open(self.bin_path, "wb") as f:
                f.write(HEADER_MAGIC)
                f.write(struct.pack("<I", HEADER_VERSION))
                f.write(struct.pack("<I", block_size))
                f.write(struct.pack("<I", dt.itemsize))
                np.array([bos_id, eos_id], dtype=dt).tofile(f)
            total_tokens = 2

        elapsed = time.monotonic() - t_start
        total_mb = (total_tokens * dt.itemsize) / (1024 * 1024)
        print(
            f"[BIN] Done: {row_count:,} rows -> {total_tokens:,} tokens "
            f"({total_mb:.1f} MB, dtype={dt}) in {elapsed:.1f}s"
        )
        if skipped > 0:
            print(f"[BIN]   Skipped {skipped:,} empty rows")
        if duplicates_skipped > 0:
            print(f"[BIN]   Skipped {duplicates_skipped:,} exact duplicate rows")
        print(f"[BIN]   Saved to {self.bin_path}")

    @staticmethod
    def _process_batch(
        rows: list[dict],
        tokenizer: Any,
        bos_id: int,
        eos_id: int,
        dt: np.dtype,
        use_batch: bool,
        seen_hashes: set[int] | None,
        n_format_workers: int,
    ) -> tuple[np.ndarray | None, int, int]:
        if n_format_workers > 1 and len(rows) >= 64:
            with ProcessPoolExecutor(max_workers=n_format_workers) as pool:
                texts = list(pool.map(_format_row_standalone, rows, chunksize=64))
        else:
            texts = [_format_row_standalone(r) for r in rows]

        skipped = 0
        duplicates_skipped = 0
        valid_texts: list[str] = []

        for text in texts:
            if not text:
                skipped += 1
                continue
            if seen_hashes is not None:
                doc_hash = int.from_bytes(
                    hashlib.md5(text.encode("utf-8")).digest(), "little"
                )
                if doc_hash in seen_hashes:
                    duplicates_skipped += 1
                    continue
                seen_hashes.add(doc_hash)
            valid_texts.append(text)

        if not valid_texts:
            return None, skipped, duplicates_skipped

        if use_batch:
            all_ids = TokenizedBinaryDataset._encode_batch(tokenizer, valid_texts)
        else:
            all_ids = [tokenizer.encode(t) for t in valid_texts]

        flat: list[int] = []
        for ids in all_ids:
            if not ids:
                ids = [bos_id, eos_id]
            else:
                if ids[0] != bos_id:
                    ids.insert(0, bos_id)
                if ids[-1] != eos_id:
                    ids.append(eos_id)
            flat.extend(ids)

        return np.array(flat, dtype=dt), skipped, duplicates_skipped

    @staticmethod
    def _log_progress(
        row_count: int,
        total_tokens: int,
        buffer_len: int,
        dt: np.dtype,
        t_start: float,
    ) -> None:
        elapsed = time.monotonic() - t_start
        rps = row_count / elapsed if elapsed > 0 else 0
        tps = (total_tokens + buffer_len) / elapsed if elapsed > 0 else 0
        disk_mb = (total_tokens * dt.itemsize) / (1024 * 1024)
        print(
            f"[BIN] {row_count:>8,} rows | "
            f"{total_tokens + buffer_len:>12,} tokens | "
            f"{disk_mb:>7.1f} MB flushed | "
            f"{rps:>6.0f} rows/s | "
            f"{tps:>8.0f} tok/s | "
            f"{elapsed:>6.1f}s elapsed"
        )

    @staticmethod
    def _iter_arrow_shards(
        arrow_files: list[Path], max_rows: Optional[int]
    ) -> Any:
        from datasets import Dataset as HFDataset

        n_shards = len(arrow_files)
        count = 0
        for shard_idx, shard_path in enumerate(arrow_files, 1):
            print(f"[BIN] Loading shard {shard_idx}/{n_shards}: {shard_path.name}")
            shard = HFDataset.from_file(str(shard_path))
            shard_len = len(shard)
            for row in shard:
                yield row
                count += 1
                if max_rows is not None and count >= max_rows:
                    del shard
                    return
            del shard
            print(f"[BIN] Shard {shard_idx}/{n_shards} done ({shard_len:,} rows, {count:,} total)")

    @staticmethod
    def _load_rows(
        source_path: Path,
        subset: Optional[str],
        split: str,
        max_rows: Optional[int],
    ) -> Any:
        from datasets import DownloadConfig, load_dataset, load_from_disk

        if source_path.exists():
            state_json = source_path / "state.json"
            arrow_files = sorted(source_path.glob("*.arrow"))

            if state_json.exists():
                ds = load_from_disk(str(source_path))
                if max_rows is not None:
                    ds = ds.select(range(min(max_rows, len(ds))))
                return ds
            elif arrow_files:
                return TokenizedBinaryDataset._iter_arrow_shards(arrow_files, max_rows)
            else:
                raise FileNotFoundError(
                    f"Path {source_path} exists but contains no Arrow dataset. "
                    "Expected either a save_to_disk format (state.json) or .arrow shard files."
                )

        kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if subset is not None:
            kwargs["name"] = subset
        download_config = DownloadConfig()
        kwargs["download_config"] = download_config

        stream = load_dataset(str(source_path), **kwargs)
        if max_rows is not None:
            return islice(stream, max_rows)
        return stream

    @staticmethod
    def _resolve_token_id(
        tokenizer: Any, custom_attr: str, hf_attr: str, default: int
    ) -> int:
        value = getattr(tokenizer, custom_attr, None)
        if value is None:
            value = getattr(tokenizer, hf_attr, None)
        return int(value) if value is not None else int(default)

    def inspect(self, tokenizer: Any, num_samples: int = 3, check_duplicates: bool = False) -> None:
        print("============== Dataset Inspection ==============")
        print(f"Total tokens in binary:  {self._n_tokens:,}")
        print(f"Total samples available: {len(self):,}")
        print(f"Model block size:        {self.block_size:,}")
        print("================================================")

        if check_duplicates:
            print("\n[BIN] Scanning for duplicate non-overlapping blocks...")
            t_start = time.monotonic()
            seen_hashes: set[int] = set()
            duplicate_blocks = 0

            n_blocks = self._n_tokens // self.block_size
            for i in range(n_blocks):
                start = i * self.block_size
                end = start + self.block_size
                block = self._mmap[start:end]

                block_hash = int.from_bytes(hashlib.md5(block.tobytes()).digest(), "little")
                if block_hash in seen_hashes:
                    duplicate_blocks += 1
                else:
                    seen_hashes.add(block_hash)

            elapsed = time.monotonic() - t_start
            dup_percent = (duplicate_blocks / n_blocks * 100) if n_blocks > 0 else 0
            print(f"[BIN] Duplicate scan complete in {elapsed:.1f}s")
            print(f"[BIN] Found {duplicate_blocks:,} duplicate blocks out of {n_blocks:,} ({dup_percent:.1f}%)")

        print(f"\n--- Showing first {num_samples} samples ---")

        for i in range(min(num_samples, len(self))):
            sample = self[i]
            input_ids = sample["input_ids"]
            targets = sample["targets"]

            print(f"\n================ Sample {i} ================")
            print(f"Shapes -> input_ids: {input_ids.shape}, targets: {targets.shape}")

            decoded_text = tokenizer.decode(input_ids.tolist())
            text_preview = decoded_text

            print("\n[Decoded Text Preview]:")
            print(text_preview)

            print("\n[Shift Verification]:")
            print(f"Input IDs  (first 5): {input_ids[:5].tolist()}")
            print(f"Target IDs (first 5): {targets[:5].tolist()}")


