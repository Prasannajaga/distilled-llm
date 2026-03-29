#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("eval_tracking_dashboard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan lm_eval runs and build tracking dashboard artifacts.")
    p.add_argument("--lm-eval-root", type=str, default="newoutput/lm_eval")
    p.add_argument("--out-json", type=str, default="newoutput/lm_eval/eval_tracking_summary.json")
    p.add_argument("--out-md", type=str, default="newoutput/lm_eval/eval_tracking_dashboard.md")
    p.add_argument("--out-html", type=str, default="newoutput/lm_eval/eval_tracking_dashboard.html")
    p.add_argument("--out-csv", type=str, default="newoutput/lm_eval/eval_tracking_table.csv")
    return p.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def stage_rank(stage: str) -> int:
    s = (stage or "").strip().lower()
    order = {
        "before-distill": 0,
        "teacher-sft": 1,
        "teacher-refine": 2,
        "after-distill": 3,
        "student-distill": 3,
        "adhoc": 9,
    }
    return order.get(s, 8)


def parse_ts(utc_text: str) -> datetime:
    if not utc_text:
        return datetime.min


def parse_teacher_config(eval_name: str) -> dict[str, Any]:
    name = str(eval_name or "")
    out: dict[str, Any] = {
        "cycle": None,
        "epochs": None,
        "learning_rate": None,
        "max_seq_length": None,
    }
    m = re.search(
        r"(?:^|-)c(?P<cycle>\d+).*?_e(?P<epochs>[0-9.]+)_lr(?P<lr>[0-9.eE+-]+)(?:_r(?P<r>\d+))?_s(?P<s>\d+)",
        name,
    )
    if not m:
        return out
    try:
        out["cycle"] = int(m.group("cycle"))
    except Exception:
        pass
    try:
        out["epochs"] = float(m.group("epochs"))
    except Exception:
        pass
    try:
        out["learning_rate"] = float(m.group("lr"))
    except Exception:
        pass
    try:
        out["max_seq_length"] = int(m.group("s"))
    except Exception:
        pass
    return out
    try:
        return datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def read_comparison(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload["_comparison_path"] = str(path)
    payload["_eval_dir"] = str(path.parent)
    return payload


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for p in sorted(root.glob("*/comparison.json")):
        row = read_comparison(p)
        if row is None:
            continue
        row.setdefault("stage", "adhoc")
        row.setdefault("experiment", "default")
        row.setdefault("run_id", "")
        row.setdefault("parent_eval", None)
        row.setdefault("notes", "")
        runs.append(row)
    runs.sort(
        key=lambda r: (
            str(r.get("experiment", "")),
            str(r.get("run_id", "")),
            stage_rank(str(r.get("stage", ""))),
            parse_ts(str(r.get("created_at_utc", ""))),
        )
    )
    return runs


def build_group_index(runs: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in runs:
        exp = str(r.get("experiment", "default"))
        rid = str(r.get("run_id", ""))
        out.setdefault(exp, {}).setdefault(rid, []).append(r)
    return out


def run_to_row(r: dict[str, Any]) -> dict[str, Any]:
    cfg = parse_teacher_config(str(r.get("eval_name", "")))
    return {
        "experiment": r.get("experiment"),
        "run_id": r.get("run_id"),
        "stage": r.get("stage"),
        "created_at_utc": r.get("created_at_utc"),
        "eval_name": r.get("eval_name"),
        "teacher_exact_match": r.get("teacher_exact_match"),
        "student_exact_match": r.get("student_exact_match"),
        "delta_teacher_minus_student": r.get("delta_teacher_minus_student"),
        "num_rows": r.get("num_rows"),
        "comparison_json": r.get("_comparison_path"),
        "notes": r.get("notes"),
        "parent_eval": r.get("parent_eval"),
        "cycle": cfg.get("cycle"),
        "epochs": cfg.get("epochs"),
        "learning_rate": cfg.get("learning_rate"),
        "max_seq_length": cfg.get("max_seq_length"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "run_id",
        "stage",
        "created_at_utc",
        "eval_name",
        "teacher_exact_match",
        "student_exact_match",
        "delta_teacher_minus_student",
        "num_rows",
        "comparison_json",
        "parent_eval",
        "notes",
        "cycle",
        "epochs",
        "learning_rate",
        "max_seq_length",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_md(path: Path, rows: list[dict[str, Any]], grouped: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Eval Tracking Dashboard")
    lines.append("")
    lines.append(f"- Total eval runs scanned: `{len(rows)}`")
    lines.append(f"- Experiments: `{len(grouped)}`")
    teacher_rows = [r for r in rows if str(r.get("stage", "")).startswith("teacher-")]
    lines.append(f"- Teacher optimization runs: `{len(teacher_rows)}`")
    lines.append("")
    if teacher_rows:
        ranked = [r for r in teacher_rows if isinstance(r.get("teacher_exact_match"), (int, float))]
        ranked.sort(key=lambda x: float(x.get("teacher_exact_match", 0.0)), reverse=True)
        best = ranked[0] if ranked else None
        if best is not None:
            lines.append("## Teacher Optimization Best")
            lines.append("")
            lines.append(f"- Best eval: `{best.get('eval_name')}`")
            lines.append(f"- Best teacher EM: `{float(best.get('teacher_exact_match')):.6f}`")
            lines.append(f"- Student EM on same eval: `{float(best.get('student_exact_match')):.6f}`")
            lines.append(f"- Delta (teacher - student): `{float(best.get('delta_teacher_minus_student')):.6f}`")
            lines.append(f"- Experiment / Run ID: `{best.get('experiment')}` / `{best.get('run_id')}`")
            lines.append("")
        lines.append("## Teacher Optimization Ranking")
        lines.append("")
        lines.append("| Rank | Eval Name | Stage | Teacher EM | Student EM | Delta | Cycle | Epochs | LR | Seq | Created UTC |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, row in enumerate(ranked, start=1):
            lines.append(
                f"| {i} | {row.get('eval_name')} | {row.get('stage')} | {float(row.get('teacher_exact_match')):.6f} | "
                f"{float(row.get('student_exact_match')):.6f} | {float(row.get('delta_teacher_minus_student')):.6f} | "
                f"{'' if row.get('cycle') is None else row.get('cycle')} | "
                f"{'' if row.get('epochs') is None else row.get('epochs')} | "
                f"{'' if row.get('learning_rate') is None else row.get('learning_rate')} | "
                f"{'' if row.get('max_seq_length') is None else row.get('max_seq_length')} | "
                f"{row.get('created_at_utc')} |"
            )
        lines.append("")

    lines.append("## Experiment Timeline")
    lines.append("")
    lines.append("| Experiment | Run ID | Stage | Eval Name | Teacher EM | Student EM | Created UTC |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for row in rows:
        t = row.get("teacher_exact_match")
        s = row.get("student_exact_match")
        t_str = f"{float(t):.6f}" if isinstance(t, (int, float)) else "n/a"
        s_str = f"{float(s):.6f}" if isinstance(s, (int, float)) else "n/a"
        lines.append(
            f"| {row.get('experiment')} | {row.get('run_id')} | {row.get('stage')} | {row.get('eval_name')} | "
            f"{t_str} | {s_str} | {row.get('created_at_utc')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tr: list[str] = []
    for row in rows:
        t = row.get("teacher_exact_match")
        s = row.get("student_exact_match")
        d = row.get("delta_teacher_minus_student")
        tr.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('experiment', '')))}</td>"
            f"<td>{html.escape(str(row.get('run_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{html.escape(str(row.get('eval_name', '')))}</td>"
            f"<td>{'' if t is None else f'{float(t):.6f}'}</td>"
            f"<td>{'' if s is None else f'{float(s):.6f}'}</td>"
            f"<td>{'' if d is None else f'{float(d):.6f}'}</td>"
            f"<td>{html.escape(str(row.get('created_at_utc', '')))}</td>"
            f"<td>{html.escape(str(row.get('comparison_json', '')))}</td>"
            "</tr>"
        )

    teacher_rows = [r for r in rows if str(r.get("stage", "")).startswith("teacher-")]
    ranked = [r for r in teacher_rows if isinstance(r.get("teacher_exact_match"), (int, float))]
    ranked.sort(key=lambda x: float(x.get("teacher_exact_match", 0.0)), reverse=True)

    teacher_tr: list[str] = []
    for i, row in enumerate(ranked, start=1):
        teacher_tr.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(str(row.get('eval_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{float(row.get('teacher_exact_match')):.6f}</td>"
            f"<td>{float(row.get('student_exact_match')):.6f}</td>"
            f"<td>{float(row.get('delta_teacher_minus_student')):.6f}</td>"
            f"<td>{'' if row.get('cycle') is None else html.escape(str(row.get('cycle')))}</td>"
            f"<td>{'' if row.get('epochs') is None else html.escape(str(row.get('epochs')))}</td>"
            f"<td>{'' if row.get('learning_rate') is None else html.escape(str(row.get('learning_rate')))}</td>"
            f"<td>{'' if row.get('max_seq_length') is None else html.escape(str(row.get('max_seq_length')))}</td>"
            f"<td>{html.escape(str(row.get('created_at_utc', '')))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval Tracking Dashboard</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif; margin: 24px; background: #f6f7fb; color: #102030; }}
    h1 {{ margin: 0 0 12px; }}
    .meta {{ margin: 0 0 16px; color: #345; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #dfe3eb; padding: 8px; font-size: 13px; text-align: left; }}
    th {{ background: #eef2f8; position: sticky; top: 0; }}
    tr:nth-child(even) td {{ background: #fbfcff; }}
  </style>
</head>
<body>
  <h1>Eval Tracking Dashboard</h1>
  <p class="meta">Runs: {len(rows)} | Teacher optimization runs: {len(teacher_rows)}</p>
  <h2>Teacher Optimization Ranking</h2>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Eval Name</th>
        <th>Stage</th>
        <th>Teacher EM</th>
        <th>Student EM</th>
        <th>Delta</th>
        <th>Cycle</th>
        <th>Epochs</th>
        <th>LR</th>
        <th>Seq</th>
        <th>Created UTC</th>
      </tr>
    </thead>
    <tbody>
      {''.join(teacher_tr)}
    </tbody>
  </table>
  <h2>All Runs Timeline</h2>
  <table>
    <thead>
      <tr>
        <th>Experiment</th>
        <th>Run ID</th>
        <th>Stage</th>
        <th>Eval Name</th>
        <th>Teacher EM</th>
        <th>Student EM</th>
        <th>Delta</th>
        <th>Created UTC</th>
        <th>comparison.json</th>
      </tr>
    </thead>
    <tbody>
      {''.join(tr)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    root = Path(args.lm_eval_root)
    runs = collect_runs(root)
    table_rows = [run_to_row(r) for r in runs]
    grouped = build_group_index(runs)

    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "lm_eval_root": str(root),
        "total_runs": len(runs),
        "experiments": len(grouped),
        "rows": table_rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(Path(args.out_csv), table_rows)
    write_md(Path(args.out_md), table_rows, grouped)
    write_html(Path(args.out_html), table_rows)

    LOGGER.info("DONE json=%s", args.out_json)
    LOGGER.info("DONE csv=%s", args.out_csv)
    LOGGER.info("DONE md=%s", args.out_md)
    LOGGER.info("DONE html=%s", args.out_html)


if __name__ == "__main__":
    main()
