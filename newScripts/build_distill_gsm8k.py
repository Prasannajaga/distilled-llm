#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("build_distill_gsm8k")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build distillation/hard-slice JSONL from lm-eval comparison.json")
    p.add_argument("--comparison-json", type=str, required=True)
    p.add_argument("--output-jsonl", type=str, required=True)
    p.add_argument(
        "--mode",
        choices=["distill_high_conf", "hard_failures", "all_gold"],
        default="distill_high_conf",
    )
    p.add_argument("--max-rows", type=int, default=0, help="<=0 means all rows.")
    return p.parse_args()


def load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_cmp = payload.get("sample_comparison", {})
    rows = sample_cmp.get("examples", [])
    if not isinstance(rows, list):
        return []
    return rows


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


def normalize_teacher_answer(raw: str, extracted: str) -> str:
    final_num = extract_priority_answer(raw) or extract_priority_answer(extracted)
    body = (raw or "").strip()
    if "####" in body:
        body = re.sub(r"####\s*[^\n\r]+", "", body).strip()
    if final_num:
        marker = f"The answer is {final_num}."
        if marker.lower() in body.lower():
            return body
        return f"{body}\n{marker}" if body else marker
    return body or extracted.strip()


def normalize_gold_answer(gold: str) -> str:
    num = extract_priority_answer(gold)
    if num:
        return f"The answer is {num}."
    return gold.strip()


def include_row(row: dict[str, Any], mode: str) -> bool:
    t_ok = bool(row.get("teacher_ok", False))
    if mode == "distill_high_conf":
        return t_ok
    if mode == "hard_failures":
        return not t_ok
    return True


def render_answer(row: dict[str, Any], mode: str) -> str:
    if mode == "distill_high_conf":
        return normalize_teacher_answer(
            str(row.get("teacher_raw", "")),
            str(row.get("teacher_pred", "")),
        )
    return normalize_gold_answer(str(row.get("gold", "")))


def main() -> None:
    configure_logging()
    args = parse_args()
    in_path = Path(args.comparison_json)
    out_path = Path(args.output_jsonl)
    rows = load_examples(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if not include_row(row, args.mode):
                continue
            question = str(row.get("question", "")).strip()
            answer = render_answer(row, args.mode).strip()
            if not question or not answer:
                continue
            out = {
                "question": question,
                "answer": answer,
                "source_idx": int(row.get("idx", kept)),
                "teacher_ok": bool(row.get("teacher_ok", False)),
                "mode": args.mode,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_rows > 0 and kept >= args.max_rows:
                break
    LOGGER.info("DONE mode=%s rows_written=%s output=%s", args.mode, kept, out_path)


if __name__ == "__main__":
    main()
