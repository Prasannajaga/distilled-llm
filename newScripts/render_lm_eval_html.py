#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ts(text: str) -> datetime:
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def collect_all_evals(root_dir: Path, per_eval_sample_limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    evals: list[dict[str, Any]] = []
    samples_by_eval: dict[str, list[dict[str, Any]]] = {}

    for comp_path in sorted(root_dir.glob("*/comparison.json")):
        try:
            payload = load_json(comp_path)
        except Exception:
            continue
        eval_name = str(payload.get("eval_name") or comp_path.parent.name)
        stage = str(payload.get("stage") or "adhoc")
        created = str(payload.get("created_at_utc") or "")
        teacher_em = payload.get("teacher_exact_match")
        student_em = payload.get("student_exact_match")
        delta = payload.get("delta_teacher_minus_student")
        evals.append(
            {
                "eval_name": eval_name,
                "stage": stage,
                "created_at_utc": created,
                "teacher_exact_match": teacher_em,
                "student_exact_match": student_em,
                "delta_teacher_minus_student": delta,
                "comparison_json": str(comp_path),
            }
        )

        sample_jsonl = comp_path.parent / "sample_columns.jsonl"
        samples_by_eval[eval_name] = _read_jsonl(sample_jsonl, max(0, int(per_eval_sample_limit)))

    evals.sort(key=lambda r: _parse_ts(str(r.get("created_at_utc", ""))), reverse=True)
    return evals, samples_by_eval


def _fmt_num(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{float(x):.6f}"
    return ""


def _row_cells(cells: list[str]) -> str:
    return "<tr>" + "".join([f"<td>{c}</td>" for c in cells]) + "</tr>"


def render_html(
    *,
    run_dir: Path,
    current: dict[str, Any],
    evals: list[dict[str, Any]],
    samples_by_eval: dict[str, list[dict[str, Any]]],
    title: str,
    row_limit: int,
) -> str:
    current_eval = str(current.get("eval_name", run_dir.name))
    examples = current.get("sample_comparison", {}).get("examples", [])
    if not isinstance(examples, list):
        examples = []
    if row_limit > 0:
        examples = examples[:row_limit]

    eval_rows: list[str] = []
    for e in evals:
        eval_rows.append(
            _row_cells(
                [
                    html.escape(str(e.get("eval_name", ""))),
                    html.escape(str(e.get("stage", ""))),
                    _fmt_num(e.get("teacher_exact_match")),
                    _fmt_num(e.get("student_exact_match")),
                    _fmt_num(e.get("delta_teacher_minus_student")),
                    html.escape(str(e.get("created_at_utc", ""))),
                ]
            )
        )

    sample_rows: list[str] = []
    for r in examples:
        sample_rows.append(
            _row_cells(
                [
                    html.escape(str(r.get("idx", ""))),
                    html.escape(str(r.get("question", ""))),
                    html.escape(str(r.get("gold", ""))),
                    html.escape(str(r.get("teacher_pred", ""))),
                    html.escape(str(r.get("student_pred", ""))),
                    "1" if bool(r.get("teacher_ok", False)) else "0",
                    "1" if bool(r.get("student_ok", False)) else "0",
                ]
            )
        )

    latest_samples = samples_by_eval.get(current_eval, [])
    latest_sample_rows: list[str] = []
    for r in latest_samples[: max(0, int(row_limit))] if row_limit > 0 else latest_samples:
        latest_sample_rows.append(
            _row_cells(
                [
                    html.escape(str(r.get("question", ""))),
                    html.escape(str(r.get("gold_answer", ""))),
                    html.escape(str(r.get("teacher_extracted", ""))),
                    html.escape(str(r.get("student_extracted", ""))),
                    html.escape(str(r.get("teacher_accuracy", ""))),
                    html.escape(str(r.get("student_accuracy", ""))),
                ]
            )
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 24px; font-family: \"IBM Plex Sans\", \"Segoe UI\", Arial, sans-serif; background: #f7f9fc; color: #102030; }}
    h1, h2 {{ margin: 0 0 10px; }}
    .card {{ background: #fff; border: 1px solid #dfe5ee; border-radius: 10px; padding: 14px; margin: 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #e4e9f1; padding: 8px; font-size: 12px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3fa; position: sticky; top: 0; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class=\"card\">
    <div><strong>Eval:</strong> <span class=\"mono\">{html.escape(current_eval)}</span></div>
    <div><strong>Stage:</strong> {html.escape(str(current.get("stage", "")))}</div>
    <div><strong>Teacher EM:</strong> {_fmt_num(current.get("teacher_exact_match"))}</div>
    <div><strong>Student EM:</strong> {_fmt_num(current.get("student_exact_match"))}</div>
    <div><strong>Delta:</strong> {_fmt_num(current.get("delta_teacher_minus_student"))}</div>
  </div>

  <div class=\"card\">
    <h2>All Evaluations</h2>
    <table>
      <thead>
        <tr>
          <th>Eval</th>
          <th>Stage</th>
          <th>Teacher EM</th>
          <th>Student EM</th>
          <th>Delta</th>
          <th>Created UTC</th>
        </tr>
      </thead>
      <tbody>
        {''.join(eval_rows)}
      </tbody>
    </table>
  </div>

  <div class=\"card\">
    <h2>Current Eval Head-to-Head</h2>
    <table>
      <thead>
        <tr>
          <th>Idx</th>
          <th>Question</th>
          <th>Gold</th>
          <th>Teacher</th>
          <th>Student</th>
          <th>T OK</th>
          <th>S OK</th>
        </tr>
      </thead>
      <tbody>
        {''.join(sample_rows)}
      </tbody>
    </table>
  </div>

  <div class=\"card\">
    <h2>Current Eval Flat Samples</h2>
    <table>
      <thead>
        <tr>
          <th>Question</th>
          <th>Gold</th>
          <th>Teacher</th>
          <th>Student</th>
          <th>T Acc</th>
          <th>S Acc</th>
        </tr>
      </thead>
      <tbody>
        {''.join(latest_sample_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
