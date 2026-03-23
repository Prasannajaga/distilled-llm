#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View teacher/student predictions side-by-side from predictions.jsonl")
    p.add_argument("predictions_file", type=str, help="Path to predictions.jsonl")
    p.add_argument("--limit", type=int, default=30, help="Rows to show (default: 30)")
    p.add_argument("--offset", type=int, default=0, help="Start offset (default: 0)")
    p.add_argument(
        "--mode",
        choices=["all", "disagree", "teacher_wrong", "student_wrong", "both_wrong", "both_correct"],
        default="all",
        help="Filter mode",
    )
    p.add_argument("--show-question", action="store_true", help="Include question text")
    p.add_argument("--max-width", type=int, default=48, help="Max width per text cell")
    return p.parse_args()


def clip(text: Any, n: int) -> str:
    s = str(text).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_match(row: dict[str, Any], mode: str) -> bool:
    t_ok = bool(row.get("teacher", {}).get("normalized_exact_match"))
    s_ok = bool(row.get("student", {}).get("normalized_exact_match"))

    if mode == "all":
        return True
    if mode == "disagree":
        return t_ok != s_ok
    if mode == "teacher_wrong":
        return not t_ok
    if mode == "student_wrong":
        return not s_ok
    if mode == "both_wrong":
        return (not t_ok) and (not s_ok)
    if mode == "both_correct":
        return t_ok and s_ok
    return True


def render_plain(rows: list[dict[str, Any]], show_question: bool, max_width: int) -> None:
    headers = ["id", "gold", "teacher", "student", "t_ok", "s_ok"]
    if show_question:
        headers.insert(1, "question")

    table: list[list[str]] = []
    for row in rows:
        t = row.get("teacher", {})
        s = row.get("student", {})
        rec = [
            str(row.get("sample_id", "")),
            clip(row.get("gold_answer", ""), max_width),
            clip(t.get("prediction", ""), max_width),
            clip(s.get("prediction", ""), max_width),
            "Y" if t.get("normalized_exact_match") else "N",
            "Y" if s.get("normalized_exact_match") else "N",
        ]
        if show_question:
            rec.insert(1, clip(row.get("question", ""), max_width))
        table.append(rec)

    widths = [len(h) for h in headers]
    for r in table:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table:
        print(fmt_row(r))


def try_render_rich(rows: list[dict[str, Any]], show_question: bool, max_width: int) -> bool:
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        return False

    table = Table(show_lines=False)
    table.add_column("id", justify="right")
    if show_question:
        table.add_column("question", overflow="fold")
    table.add_column("gold", overflow="fold")
    table.add_column("teacher", overflow="fold")
    table.add_column("student", overflow="fold")
    table.add_column("t_ok", justify="center")
    table.add_column("s_ok", justify="center")

    for row in rows:
        t = row.get("teacher", {})
        s = row.get("student", {})
        t_ok = bool(t.get("normalized_exact_match"))
        s_ok = bool(s.get("normalized_exact_match"))
        vals = [
            str(row.get("sample_id", "")),
            clip(row.get("gold_answer", ""), max_width),
            clip(t.get("prediction", ""), max_width),
            clip(s.get("prediction", ""), max_width),
            "[green]Y[/green]" if t_ok else "[red]N[/red]",
            "[green]Y[/green]" if s_ok else "[red]N[/red]",
        ]
        if show_question:
            vals.insert(1, clip(row.get("question", ""), max_width))
        table.add_row(*vals)

    Console().print(table)
    return True


def main() -> None:
    args = parse_args()
    path = Path(args.predictions_file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    all_rows = load_rows(path)
    filtered = [r for r in all_rows if is_match(r, args.mode)]

    start = max(0, args.offset)
    end = start + max(1, args.limit)
    page = filtered[start:end]

    print(f"file={path}")
    print(f"total_rows={len(all_rows)} filtered={len(filtered)} shown={len(page)} offset={start} mode={args.mode}")

    if not page:
        print("No rows for this selection.")
        return

    if not try_render_rich(page, args.show_question, args.max_width):
        render_plain(page, args.show_question, args.max_width)


if __name__ == "__main__":
    main()
