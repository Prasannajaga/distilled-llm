#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View custom lm-eval run outputs")
    p.add_argument("run_dir", type=str, help="Path like newoutput/lm_eval/custom_gsm8k_... ")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-width", type=int, default=60)
    return p.parse_args()


def clip(x: Any, n: int) -> str:
    s = str(x).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_samples_file(model_dir: Path) -> Path | None:
    matches = sorted(model_dir.rglob("samples*.jsonl"))
    return matches[0] if matches else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_text_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    doc = row.get("doc", {}) if isinstance(row.get("doc", {}), dict) else {}
    q = doc.get("question", row.get("prompt", ""))
    gold = row.get("target", doc.get("answer", ""))

    pred = ""
    if isinstance(row.get("resps"), list) and row["resps"]:
        first = row["resps"][0]
        if isinstance(first, list) and first:
            pred = first[0]
        else:
            pred = first
    if not pred and isinstance(row.get("filtered_resps"), list) and row["filtered_resps"]:
        pred = row["filtered_resps"][0]

    return str(q), str(gold), str(pred)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    comp = run_dir / "comparison.json"
    if not comp.exists():
        raise FileNotFoundError(f"comparison.json not found in {run_dir}")

    summary = load_json(comp)
    print(f"run_dir={run_dir}")
    print(f"task={summary.get('task_name')}")
    print(f"rows={summary.get('num_rows')}")
    print(f"teacher={summary.get('teacher_model')} score={summary.get('teacher_exact_match')}")
    print(f"student={summary.get('student_model')} score={summary.get('student_exact_match')}")
    print(f"delta={summary.get('delta_teacher_minus_student')}")

    teacher_dir = run_dir / "teacher"
    student_dir = run_dir / "student"
    t_samples = find_samples_file(teacher_dir)
    s_samples = find_samples_file(student_dir)
    if not t_samples or not s_samples:
        print("Sample JSONL files not found. Ensure run used --log_samples.")
        return

    t_rows = read_jsonl(t_samples)
    s_rows = read_jsonl(s_samples)
    n = min(len(t_rows), len(s_rows), max(1, args.limit))

    print(f"\nteacher_samples={t_samples}")
    print(f"student_samples={s_samples}")
    print(f"showing={n}")

    headers = ["idx", "question", "gold", "teacher", "student"]
    widths = [5, args.max_width, args.max_width, args.max_width, args.max_width]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))

    for i in range(n):
        tq, tg, tp = extract_text_fields(t_rows[i])
        sq, sg, sp = extract_text_fields(s_rows[i])
        q = tq or sq
        g = tg or sg
        cols = [
            str(i),
            clip(q, widths[1]),
            clip(g, widths[2]),
            clip(tp, widths[3]),
            clip(sp, widths[4]),
        ]
        print(" | ".join(cols[j].ljust(widths[j]) for j in range(len(cols))))


if __name__ == "__main__":
    main()
