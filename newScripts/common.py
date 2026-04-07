#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import hashlib
import re
import random
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
IDX_FIELD = "idx"
GOLD_FIELD = "gold"
TEACHER_PRED_FIELD = "teacher_pred"
STUDENT_PRED_FIELD = "student_pred"
TEACHER_RAW_FIELD = "teacher_raw"
STUDENT_RAW_FIELD = "student_raw"
TEACHER_OK_FIELD = "teacher_ok"
STUDENT_OK_FIELD = "student_ok"
WINNER_FIELD = "winner"

SAMPLE_COMPARISON_KEY = "sample_comparison"
EXAMPLES_KEY = "examples"

GOLD_ANSWER_FIELD = "gold_answer"
TEACHER_EXTRACTED_FIELD = "teacher_extracted"
TEACHER_RAW_PREDICTION_FIELD = "teacher_raw_prediction"
STUDENT_EXTRACTED_FIELD = "student_extracted"
STUDENT_RAW_PREDICTION_FIELD = "student_raw_prediction"
TEACHER_MATCH_FIELD = "teacher_match"
STUDENT_MATCH_FIELD = "student_match"
WINNER_LABEL_FIELD = "winner_label"

FORMAT_ANSWER_ONLY = "answer_only"
FORMAT_SHORT_RATIONALE = "short_rationale"
FORMAT_FULL_RATIONALE = "full_rationale"
TARGET_FORMATS: tuple[str, ...] = (
    FORMAT_ANSWER_ONLY,
    FORMAT_SHORT_RATIONALE,
    FORMAT_FULL_RATIONALE,
)

TARGET_SOURCE_TEACHER = "teacher"
TARGET_SOURCE_GOLD = "gold"

FINAL_ANSWER_MARKER_TEMPLATE = "The answer is <number>."
ANSWER_CONTRACT_INSTRUCTION = "Solve step by step and end with 'The answer is <number>.'"

BUCKET_TEACHER_CORRECT_STUDENT_WRONG = "teacher correct / student wrong"
BUCKET_TEACHER_WRONG_STUDENT_CORRECT = "teacher wrong / student correct"
BUCKET_BOTH_CORRECT = "both correct"
BUCKET_BOTH_WRONG = "both wrong"

TAIL_LINES = 25
HEARTBEAT_SECONDS = 30


@dataclass(frozen=True)
class PhasePaths:
    phase_root: Path
    datasets_root: Path
    manifests_root: Path
    models_root: Path
    reports_root: Path

    def ensure(self) -> None:
        for path in (
            self.phase_root,
            self.datasets_root,
            self.manifests_root,
            self.models_root,
            self.reports_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)


def log_state(logger: logging.Logger, state: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={fields[k]}" for k in sorted(fields))
    if payload:
        logger.info("STATE | %s | %s", state, payload)
    else:
        logger.info("STATE | %s", state)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "run"


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> Path:
    p = ensure_parent_dir(Path(path))
    p.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return p


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    p = ensure_parent_dir(Path(path))
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return p


def build_phase_paths(output_root: str | Path, phase_slug: str) -> PhasePaths:
    phase_root = Path(output_root) / phase_slug
    return PhasePaths(
        phase_root=phase_root,
        datasets_root=phase_root / "datasets",
        manifests_root=phase_root / "manifests",
        models_root=phase_root / "models",
        reports_root=phase_root / "reports",
    )


def run_logged_command(
    cmd: Sequence[str],
    *,
    logger: logging.Logger,
    dry_run: bool,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    tail_lines: int = TAIL_LINES,
    heartbeat_seconds: int = HEARTBEAT_SECONDS,
) -> None:
    logger.info("CMD: %s", shlex.join(list(cmd)))
    if dry_run:
        logger.info("Dry run mode: command skipped.")
        return

    start = time.perf_counter()
    last_heartbeat = start
    tail: list[str] = []
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            msg = line.rstrip("\n")
            tail.append(msg)
            if len(tail) > tail_lines:
                del tail[:-tail_lines]

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    while proc.poll() is None:
        time.sleep(1.0)
        now = time.perf_counter()
        if (now - last_heartbeat) >= heartbeat_seconds:
            logger.info("Still running (%.1fs): %s", now - start, tail[-1][:200] if tail else "(no output yet)")
            last_heartbeat = now
    thread.join(timeout=5)
    rc = proc.wait()
    if rc != 0:
        if tail:
            logger.error("Last %d output lines:\n%s", len(tail), "\n".join(tail))
        raise subprocess.CalledProcessError(rc, list(cmd))
    logger.info("Command succeeded in %.2fs", time.perf_counter() - start)


def safe_int(x: Any, fallback: int) -> int:
    try:
        return int(x)
    except Exception:
        return fallback


def extract_priority_answer(text: str) -> str:
    if not text:
        return ""
    marker = re.search(r"(?is)the answer is\s*(-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)", text)
    if marker:
        return marker.group(1).replace(",", "").strip()
    hash_marker = re.search(r"####\s*(-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*)", text)
    if hash_marker:
        return hash_marker.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_terminal_answer_markers(text: str) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    body = re.sub(r"(?is)\n?\s*the answer is\s*-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*\.?\s*$", "", body).strip()
    body = re.sub(r"(?is)\n?\s*####\s*-?[0-9]+(?:\.[0-9]+)?(?:,[0-9]{3})*\s*$", "", body).strip()
    return body


def final_answer_marker(final_num: str) -> str:
    return f"The answer is {final_num}." if final_num else ""


def build_short_rationale(body: str, marker: str, max_sentences: int, max_chars: int) -> str:
    if not body:
        return marker
    flat = normalize_ws(body)
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", flat) if c.strip()]
    if not chunks:
        chunks = [flat]
    selected: list[str] = []
    for chunk in chunks:
        selected.append(chunk)
        if max_sentences > 0 and len(selected) >= max_sentences:
            break
    short = " ".join(selected).strip()
    if max_chars > 0 and len(short) > max_chars:
        short = short[: max(1, max_chars)].rstrip()
        short = re.sub(r"\s+\S*$", "", short).strip() or short
    if marker and marker.lower() not in short.lower():
        short = f"{short}\n{marker}" if short else marker
    return short.strip()


def render_target_answer(
    row: Mapping[str, Any],
    *,
    target_format: str,
    target_source: str,
    short_max_sentences: int,
    short_max_chars: int,
) -> str:
    teacher_raw = str(row.get(TEACHER_RAW_FIELD, "") or "")
    teacher_pred = str(row.get(TEACHER_PRED_FIELD, "") or "")
    gold = str(row.get(GOLD_FIELD, "") or "")

    source_text = teacher_raw if target_source == TARGET_SOURCE_TEACHER else gold
    if target_source == TARGET_SOURCE_GOLD:
        final_num = (
            extract_priority_answer(gold)
            or extract_priority_answer(teacher_pred)
            or extract_priority_answer(teacher_raw)
        )
    else:
        final_num = (
            extract_priority_answer(teacher_pred)
            or extract_priority_answer(teacher_raw)
            or extract_priority_answer(gold)
        )
    marker = final_answer_marker(final_num)

    if target_format == FORMAT_ANSWER_ONLY:
        return marker or normalize_ws(source_text)

    body = strip_terminal_answer_markers(source_text)
    if target_format == FORMAT_SHORT_RATIONALE:
        return build_short_rationale(body, marker, short_max_sentences, short_max_chars)

    if marker and marker.lower() not in body.lower():
        return f"{body}\n{marker}".strip() if body else marker
    return (body or marker).strip()


def compute_bucket_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    both_correct = 0
    both_wrong = 0
    teacher_only = 0
    student_only = 0
    for row in rows:
        teacher_ok = bool(row.get(TEACHER_OK_FIELD, False))
        student_ok = bool(row.get(STUDENT_OK_FIELD, False))
        if teacher_ok and student_ok:
            both_correct += 1
        elif teacher_ok and not student_ok:
            teacher_only += 1
        elif (not teacher_ok) and student_ok:
            student_only += 1
        else:
            both_wrong += 1
    return {
        BUCKET_TEACHER_CORRECT_STUDENT_WRONG: teacher_only,
        BUCKET_TEACHER_WRONG_STUDENT_CORRECT: student_only,
        BUCKET_BOTH_CORRECT: both_correct,
        BUCKET_BOTH_WRONG: both_wrong,
    }


def bucket_key_from_flags(teacher_ok: bool, student_ok: bool) -> str:
    if teacher_ok and not student_ok:
        return BUCKET_TEACHER_CORRECT_STUDENT_WRONG
    if (not teacher_ok) and student_ok:
        return BUCKET_TEACHER_WRONG_STUDENT_CORRECT
    if teacher_ok and student_ok:
        return BUCKET_BOTH_CORRECT
    return BUCKET_BOTH_WRONG


def compute_bucket_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = compute_bucket_counts(rows)
    total = int(sum(counts.values()))
    teacher_correct = int(counts.get(BUCKET_TEACHER_CORRECT_STUDENT_WRONG, 0) + counts.get(BUCKET_BOTH_CORRECT, 0))
    student_correct = int(counts.get(BUCKET_TEACHER_WRONG_STUDENT_CORRECT, 0) + counts.get(BUCKET_BOTH_CORRECT, 0))
    teacher_accuracy = (teacher_correct / total) if total > 0 else None
    student_accuracy = (student_correct / total) if total > 0 else None
    return {
        "total_rows": total,
        "bucket_counts": counts,
        "teacher_accuracy": teacher_accuracy,
        "student_accuracy": student_accuracy,
        "delta_teacher_minus_student": (
            (teacher_accuracy - student_accuracy)
            if isinstance(teacher_accuracy, float) and isinstance(student_accuracy, float)
            else None
        ),
        "recovery_count_teacher_correct_student_wrong": int(counts.get(BUCKET_TEACHER_CORRECT_STUDENT_WRONG, 0)),
        "damage_count_both_correct": int(counts.get(BUCKET_BOTH_CORRECT, 0)),
        "damage_count_teacher_wrong_student_correct": int(counts.get(BUCKET_TEACHER_WRONG_STUDENT_CORRECT, 0)),
    }


def filter_teacher_correct_student_wrong(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    kept = [
        dict(row)
        for row in rows
        if bool(row.get(TEACHER_OK_FIELD, False)) and (not bool(row.get(STUDENT_OK_FIELD, False)))
    ]
    if max_rows > 0:
        kept = kept[:max_rows]
    return kept


def normalize_question_text(question: str) -> str:
    text = str(question or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def question_hash(question: str) -> str:
    norm = normalize_question_text(question)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _extract_row_id(row: Mapping[str, Any], id_fields: Sequence[str]) -> str:
    for field in id_fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def build_example_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_fields: Sequence[str] = ("sample_id", "id", "idx"),
    question_field: str = QUESTION_FIELD,
) -> list[str]:
    id_values: list[str] = []
    for row in rows:
        id_values.append(_extract_row_id(row, id_fields))
    non_empty_ids = [x for x in id_values if x]
    id_reliable = len(non_empty_ids) > 0 and len(set(non_empty_ids)) == len(non_empty_ids)

    keys: list[str] = []
    for idx, row in enumerate(rows):
        row_id = id_values[idx] if idx < len(id_values) else ""
        question = str(row.get(question_field, "") or "")
        if id_reliable and row_id:
            keys.append(f"id:{row_id}")
            continue
        q_hash = question_hash(question)
        keys.append(f"qhash:{q_hash}")
    return keys


def compute_overlap_report(named_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    key_sets: dict[str, set[str]] = {}
    for split_name, rows in named_rows.items():
        preferred_keys = build_example_keys(rows)
        qhash_only_keys = build_example_keys(rows, id_fields=())
        key_set = set(preferred_keys).union(set(qhash_only_keys))
        split_counts[str(split_name)] = len(key_set)
        key_sets[str(split_name)] = key_set

    pairwise: list[dict[str, Any]] = []
    split_names = list(key_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            left = split_names[i]
            right = split_names[j]
            left_set = key_sets[left]
            right_set = key_sets[right]
            overlap = left_set.intersection(right_set)
            left_count = len(left_set)
            right_count = len(right_set)
            overlap_count = len(overlap)
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "left_count": left_count,
                    "right_count": right_count,
                    "overlap_count": overlap_count,
                    "overlap_pct_of_left": (overlap_count / left_count) if left_count > 0 else None,
                    "overlap_pct_of_right": (overlap_count / right_count) if right_count > 0 else None,
                    "sample_overlap_keys": sorted(list(overlap))[:10],
                }
            )

    max_overlap_pct = 0.0
    for item in pairwise:
        for key in ("overlap_pct_of_left", "overlap_pct_of_right"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                max_overlap_pct = max(max_overlap_pct, float(value))
    return {
        "generated_at_utc": utc_now(),
        "split_counts": split_counts,
        "pairwise": pairwise,
        "max_overlap_pct": max_overlap_pct,
    }


def split_rows_train_val(
    rows: Sequence[Mapping[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    ratio = float(val_ratio)
    if ratio <= 0.0:
        all_rows = [dict(row) for row in rows]
        all_idx = list(range(len(all_rows)))
        return all_rows, [], all_idx, []
    if ratio >= 1.0:
        raise ValueError("val_ratio must be < 1.0")
    total = len(rows)
    if total < 2:
        raise ValueError("Need at least 2 rows for train/val split.")
    val_size = int(round(total * ratio))
    val_size = max(1, min(total - 1, val_size))
    indices = list(range(total))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    train_rows = [dict(rows[i]) for i in train_indices]
    val_rows = [dict(rows[i]) for i in val_indices]
    return train_rows, val_rows, train_indices, val_indices


def render_dataset_rows(
    filtered_rows: Sequence[Mapping[str, Any]],
    *,
    target_format: str,
    target_source: str,
    short_max_sentences: int,
    short_max_chars: int,
    max_answer_chars: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    output_rows: list[dict[str, Any]] = []
    kept_ids: list[int] = []
    for i, row in enumerate(filtered_rows):
        question = str(row.get(QUESTION_FIELD, "")).strip()
        answer = render_target_answer(
            row,
            target_format=target_format,
            target_source=target_source,
            short_max_sentences=short_max_sentences,
            short_max_chars=short_max_chars,
        ).strip()
        if not question or not answer:
            continue
        if max_answer_chars > 0 and len(answer) > max_answer_chars:
            continue
        output_rows.append({QUESTION_FIELD: question, ANSWER_FIELD: answer})
        kept_ids.append(safe_int(row.get(IDX_FIELD, i), i))
    return output_rows, kept_ids


def norm_text_for_match(text: str) -> str:
    preferred = extract_priority_answer(text)
    normalized = preferred if preferred else text.strip().lower()
    marker = re.search(r"####\s*([^\n\r]+)", normalized)
    if marker:
        normalized = marker.group(1).strip()
    normalized = normalized.replace(",", "").replace("$", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", text)
    return nums[-1] if nums else None


def answers_match(pred: str, gold: str) -> bool:
    pred_norm = norm_text_for_match(pred)
    gold_norm = norm_text_for_match(gold)
    if not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    pred_num = extract_last_number(pred_norm)
    gold_num = extract_last_number(gold_norm)
    return pred_num is not None and gold_num is not None and pred_num == gold_num
