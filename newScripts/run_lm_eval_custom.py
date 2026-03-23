#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

DEFAULT_STUDENT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2-Math-1.5B-Instruct"
DEFAULT_GSM8K_PATH = "/media/prasanna/716F26140AED9B67/datasets/GSM8K"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run custom lm-eval benchmark on local GSM8K-style dataset")
    p.add_argument("--dataset-path", type=str, default=DEFAULT_GSM8K_PATH)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    p.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=str, default="1")
    p.add_argument("--limit", type=str, default=None, help="lm-eval limit, e.g. 100 or 0.1")
    p.add_argument("--num-fewshot", type=int, default=8)
    p.add_argument("--gen-max-toks", type=int, default=64)
    p.add_argument(
        "--sample-comparison-limit",
        type=int,
        default=0,
        help="How many sample comparison rows to store in comparison.json (<=0 means all).",
    )
    p.add_argument(
        "--local-models-only",
        action="store_true",
        help="Force local model loading only (no Hugging Face download).",
    )
    p.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Run name for output folder and task naming (no timestamp suffix auto-added).",
    )
    p.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite existing output dir (keeps stable naming; no auto timestamp suffix).",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "eval"


def ensure_unique_dir(base_root: Path, desired_name: str) -> Path:
    candidate = base_root / desired_name
    if not candidate.exists():
        return candidate
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base_root / f"{desired_name}__{ts}"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        c = base_root / f"{desired_name}__{ts}_{idx}"
        if not c.exists():
            return c
        idx += 1


def taskify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "eval"


def _search_arrow(root: Path, split: str) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    target_name = f"gsm8k-{split}.arrow"
    direct = root / target_name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(target_name))
    return matches[0] if matches else None


def resolve_split_arrow(dataset_path: str, split: str) -> Path:
    requested = Path(dataset_path)
    target_name = f"gsm8k-{split}.arrow"

    if requested.exists():
        if requested.is_file():
            if requested.name != target_name:
                raise FileNotFoundError(
                    f"Expected file name {target_name} for split '{split}', got: {requested}"
                )
            return requested
        found = _search_arrow(requested, split)
        if found is not None:
            return found

    fallback_roots = [
        Path("/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k"),
        Path("/media/prasanna/716F26140AED9B67/datasets/openai___gsm8k/main"),
    ]
    if requested.name.lower() == "gsm8k":
        fallback_roots = [requested.parent / "openai___gsm8k", requested.parent / "openai___gsm8k" / "main"] + fallback_roots

    for root in fallback_roots:
        found = _search_arrow(root, split)
        if found is not None:
            return found

    raise FileNotFoundError(
        f"Could not locate {target_name}. Checked {requested} and standard openai___gsm8k fallbacks."
    )


def to_jsonl_from_arrow(split_arrow: Path, out_jsonl: Path, max_samples: int) -> int:
    ds = Dataset.from_file(str(split_arrow))
    count = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in ds:
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not q or not a:
                continue
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            count += 1
            if max_samples > 0 and count >= max_samples:
                break
    return count


def write_task_yaml(task_path: Path, data_jsonl: Path, num_fewshot: int, task_name: str) -> str:
    yaml = f"""task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data_jsonl.as_posix()}
output_type: generate_until
test_split: test
training_split: null
validation_split: null
doc_to_text: |
  Q: {{{{question}}}}

  A:
doc_to_target: "{{{{answer.split('####')[-1].strip()}}}}"
num_fewshot: {num_fewshot}
filter_list:
  - name: strict-match
    filter:
      - function: regex
        regex_pattern: 'The answer is (-?[0-9.,]+).'
      - function: take_first
  - name: flexible-extract
    filter:
      - function: regex
        group_select: -1
        regex_pattern: '(-?[$0-9.,]{{2,}})|(-?[0-9]+)'
      - function: take_first
generation_kwargs:
  do_sample: false
  until:
    - "Q:"
    - "</s>"
    - "<|im_end|>"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - '\\$'
      - '(?s).*#### '
      - '\\.$'
metadata:
  version: 1.0
"""
    task_path.write_text(yaml, encoding="utf-8")
    return task_name


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def find_latest_results_json(output_root: Path) -> Path | None:
    candidates = sorted(
        output_root.rglob("results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_sample_rows(samples_jsonl: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with samples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and limit > 0 and len(rows) >= limit:
                break
    return rows


def find_latest_samples_jsonl(output_root: Path) -> Path | None:
    candidates = sorted(
        output_root.rglob("samples*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def materialize_readable_artifacts(model_out_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Copies latest lm-eval raw outputs to stable readable paths:
      - <model_out_dir>/results.json
      - <model_out_dir>/samples.jsonl
    """
    latest_results = find_latest_results_json(model_out_dir)
    latest_samples = find_latest_samples_jsonl(model_out_dir)

    stable_results: Path | None = None
    stable_samples: Path | None = None
    if latest_results is not None and latest_results.exists():
        stable_results = model_out_dir / "results.json"
        shutil.copy2(latest_results, stable_results)
    if latest_samples is not None and latest_samples.exists():
        stable_samples = model_out_dir / "samples.jsonl"
        shutil.copy2(latest_samples, stable_samples)
    return stable_results, stable_samples


def _filter_rank(filter_name: Any) -> int:
    name = str(filter_name or "").strip().lower()
    if name == "flexible-extract":
        return 0
    if name == "strict-match":
        return 1
    return 10


def _sample_group_key(row: dict[str, Any]) -> str:
    for k in ("doc_hash", "prompt_hash", "target_hash"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return f"{k}:{v}"
    return f"doc_id:{row.get('doc_id')}"


def _is_invalid_filtered(row: dict[str, Any]) -> bool:
    fr = row.get("filtered_resps")
    if isinstance(fr, list) and fr:
        return str(fr[0]).strip().lower() == "[invalid]"
    return False


def read_best_sample_rows(path: Path, limit_docs: int) -> list[dict[str, Any]]:
    best_by_doc: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _sample_group_key(row)
            prev = best_by_doc.get(key)
            if prev is None:
                best_by_doc[key] = row
            else:
                prev_rank = (_filter_rank(prev.get("filter")), 1 if _is_invalid_filtered(prev) else 0)
                row_rank = (_filter_rank(row.get("filter")), 1 if _is_invalid_filtered(row) else 0)
                if row_rank < prev_rank:
                    best_by_doc[key] = row

    rows = list(best_by_doc.values())
    rows.sort(key=lambda r: int(r.get("doc_id", 10**9)) if str(r.get("doc_id", "")).isdigit() else 10**9)
    if limit_docs > 0:
        rows = rows[:limit_docs]
    return rows


def _extract_triplet(sample_row: dict[str, Any]) -> tuple[str, str, str, str]:
    doc = sample_row.get("doc", {}) if isinstance(sample_row.get("doc", {}), dict) else {}
    question = str(doc.get("question", sample_row.get("prompt", "")))
    gold = str(sample_row.get("target", doc.get("answer", "")))

    raw_pred = ""
    resps = sample_row.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            raw_pred = str(first[0])
        else:
            raw_pred = str(first)

    extracted_pred = ""
    fresps = sample_row.get("filtered_resps")
    if isinstance(fresps, list) and fresps:
        extracted_pred = str(fresps[0])
    if extracted_pred.strip().lower() == "[invalid]":
        extracted_pred = ""
    if not extracted_pred:
        extracted_pred = raw_pred
    return question, gold, extracted_pred, raw_pred


def _norm_text(x: str) -> str:
    s = x.strip().lower()
    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        s = m.group(1).strip()
    s = s.replace(",", "").replace("$", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_num(x: str) -> str | None:
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", x)
    return nums[-1] if nums else None


def _matches(pred: str, gold: str) -> bool:
    p = _norm_text(pred)
    g = _norm_text(gold)
    if not g:
        return False
    if p == g:
        return True
    pn = _extract_num(p)
    gn = _extract_num(g)
    return pn is not None and gn is not None and pn == gn


def build_sample_comparison(
    teacher_samples: Path | None,
    student_samples: Path | None,
    limit: int = 80,
) -> dict[str, Any]:
    if teacher_samples is None or student_samples is None:
        return {
            "teacher_samples_file": str(teacher_samples) if teacher_samples else None,
            "student_samples_file": str(student_samples) if student_samples else None,
            "loaded_rows": 0,
            "teacher_wins": 0,
            "student_wins": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "examples": [],
        }

    t_rows = read_best_sample_rows(teacher_samples, limit_docs=limit)
    s_rows = read_best_sample_rows(student_samples, limit_docs=limit)
    n = min(len(t_rows), len(s_rows))

    teacher_wins = 0
    student_wins = 0
    both_correct = 0
    both_wrong = 0
    examples: list[dict[str, Any]] = []

    for i in range(n):
        tq, tg, tp, tp_raw = _extract_triplet(t_rows[i])
        sq, sg, sp, sp_raw = _extract_triplet(s_rows[i])
        question = tq or sq
        gold = tg or sg

        t_ok = _matches(tp, gold)
        s_ok = _matches(sp, gold)

        winner = "none"
        if t_ok and not s_ok:
            teacher_wins += 1
            winner = "teacher"
        elif s_ok and not t_ok:
            student_wins += 1
            winner = "student"
        elif t_ok and s_ok:
            both_correct += 1
            winner = "both"
        else:
            both_wrong += 1

        examples.append(
            {
                "idx": i,
                "question": question,
                "gold": gold,
                "teacher_pred": tp,
                "student_pred": sp,
                "teacher_pred_extracted": tp,
                "student_pred_extracted": sp,
                "teacher_pred_raw": tp_raw,
                "student_pred_raw": sp_raw,
                "teacher_ok": t_ok,
                "student_ok": s_ok,
                "winner": winner,
            }
        )

    return {
        "teacher_samples_file": str(teacher_samples),
        "student_samples_file": str(student_samples),
        "loaded_rows": n,
        "teacher_wins": teacher_wins,
        "student_wins": student_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "examples": examples,
    }


def extract_score(result_json: Path, task_name: str) -> tuple[float | None, str | None]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    task = results.get(task_name, {})
    if not isinstance(task, dict):
        return None, None

    preferred = ("exact_match,flexible-extract", "exact_match")
    for key in preferred:
        val = task.get(key)
        if isinstance(val, (int, float)):
            return float(val), key

    fallback_key: str | None = None
    fallback_val: float | None = None
    for k, v in task.items():
        if "exact_match" in k and "stderr" not in k and isinstance(v, (int, float)):
            fallback_key = k
            fallback_val = float(v)
            break
    return fallback_val, fallback_key


def update_leaderboard(root_dir: Path, summary: dict[str, Any]) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    lb_jsonl = root_dir / "leaderboard.jsonl"
    rows: list[dict[str, Any]] = []
    if lb_jsonl.exists():
        with lb_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.append(summary)

    with lb_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sortable = [
        r
        for r in rows
        if isinstance(r.get("teacher_exact_match"), (int, float)) and isinstance(r.get("student_exact_match"), (int, float))
    ]
    sortable.sort(key=lambda r: float(r.get("teacher_exact_match", 0.0)), reverse=True)

    md = []
    md.append("# lm-eval Leaderboard")
    md.append("")
    md.append("| Rank | Eval Name | Created (UTC) | Rows | Teacher EM | Student EM | Delta |")
    md.append("|---:|---|---|---:|---:|---:|---:|")
    for i, r in enumerate(sortable, start=1):
        md.append(
            f"| {i} | {r.get('eval_name')} | {r.get('created_at_utc')} | {r.get('num_rows')} | "
            f"{float(r.get('teacher_exact_match')):.6f} | {float(r.get('student_exact_match')):.6f} | "
            f"{float(r.get('delta_teacher_minus_student', 0.0)):.6f} |"
        )
    (root_dir / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")

    cfg = {
        "schema_version": 1,
        "primary_metric": "teacher_exact_match",
        "secondary_metric": "delta_teacher_minus_student",
        "sort_order": "desc",
        "registry_file": str(lb_jsonl),
    }
    (root_dir / "leaderboard_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def resolve_model_ref(model_ref: str, local_only: bool) -> tuple[str, bool]:
    """
    Returns:
      - model_ref_for_lm_eval (absolute local path if directory exists, otherwise original string)
      - is_local_path
    """
    p = Path(model_ref).expanduser()
    if p.exists():
        p = p.resolve()
        # Handle Hugging Face cache-root layout:
        #   models--ORG--NAME/{blobs,refs,snapshots/<rev>/...}
        # Transformers expects the snapshot dir that contains config.json.
        if p.is_dir() and (p / "snapshots").is_dir() and (p / "refs").is_dir():
            snap_root = p / "snapshots"
            chosen: Path | None = None
            ref_main = p / "refs" / "main"
            if ref_main.exists():
                rev = ref_main.read_text(encoding="utf-8").strip()
                cand = snap_root / rev
                if cand.is_dir():
                    chosen = cand
            if chosen is None:
                snaps = sorted([d for d in snap_root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
                if snaps:
                    chosen = snaps[0]
            if chosen is None:
                raise FileNotFoundError(f"No snapshot directories found in HF cache path: {p}")
            p = chosen

        if p.is_dir() and not (p / "config.json").exists():
            raise FileNotFoundError(
                f"Local model directory does not contain config.json: {p}"
            )

        if p.is_dir():
            has_weights = any((p / n).exists() for n in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
            if not has_weights:
                raise FileNotFoundError(
                    f"Local model directory found but no weights file present: {p}"
                )

        return str(p), True
    if local_only:
        raise FileNotFoundError(
            f"--local-models-only was set, but model path does not exist locally: {model_ref}"
        )
    return model_ref, False


def main() -> None:
    args = parse_args()
    base_root = Path("newoutput") / "lm_eval"
    eval_name_raw = args.eval_name or "custom_gsm8k"
    eval_name = slugify(eval_name_raw)
    out_dir = Path(args.output_dir) if args.output_dir else (base_root / eval_name)
    eval_name_final = out_dir.name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {out_dir}\n"
            "Use --overwrite-output to reuse stable directory naming."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    split_arrow = resolve_split_arrow(args.dataset_path, args.split)
    task_slug = taskify(eval_name_final)
    task_name = f"gsm8k_local_{task_slug}"
    data_jsonl = out_dir / f"{task_name}_{args.split}.jsonl"
    n_rows = to_jsonl_from_arrow(split_arrow, data_jsonl, args.max_samples)

    task_yaml = out_dir / f"{task_name}.yaml"
    task_name = write_task_yaml(task_yaml, data_jsonl, args.num_fewshot, task_name)

    teacher_ref, teacher_is_local = resolve_model_ref(args.teacher_model, args.local_models_only)
    student_ref, student_is_local = resolve_model_ref(args.student_model, args.local_models_only)

    common = [
        "uv",
        "run",
        "python",
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--tasks",
        task_name,
        "--include_path",
        str(out_dir),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--log_samples",
        "--gen_kwargs",
        f"do_sample=False,temperature=0.0,max_gen_toks={int(args.gen_max_toks)}",
    ]
    if args.limit:
        common.extend(["--limit", str(args.limit)])

    teacher_out = out_dir / "teacher"
    student_out = out_dir / "student"

    teacher_model_args = [f"pretrained={teacher_ref}", "trust_remote_code=True", "dtype=auto"]
    student_model_args = [f"pretrained={student_ref}", "trust_remote_code=True", "dtype=auto"]
    if args.local_models_only:
        teacher_model_args.append("local_files_only=True")
        student_model_args.append("local_files_only=True")

    teacher_cmd = common + [
        "--model_args",
        ",".join(teacher_model_args),
        "--output_path",
        str(teacher_out),
    ]
    student_cmd = common + [
        "--model_args",
        ",".join(student_model_args),
        "--output_path",
        str(student_out),
    ]

    if args.dry_run:
        print("[DRY] teacher:", " ".join(teacher_cmd))
        print("[DRY] student:", " ".join(student_cmd))
        return

    run_cmd(teacher_cmd, Path.cwd())
    run_cmd(student_cmd, Path.cwd())

    teacher_res_stable, teacher_samples_stable = materialize_readable_artifacts(teacher_out)
    student_res_stable, student_samples_stable = materialize_readable_artifacts(student_out)

    teacher_res = teacher_res_stable if teacher_res_stable else find_latest_results_json(teacher_out)
    student_res = student_res_stable if student_res_stable else find_latest_results_json(student_out)
    t_score, t_metric_key = extract_score(teacher_res, task_name) if teacher_res else (None, None)
    s_score, s_metric_key = extract_score(student_res, task_name) if student_res else (None, None)

    teacher_samples = teacher_samples_stable if teacher_samples_stable else find_latest_samples_jsonl(teacher_out)
    student_samples = student_samples_stable if student_samples_stable else find_latest_samples_jsonl(student_out)
    sample_cmp = build_sample_comparison(
        teacher_samples,
        student_samples,
        limit=int(args.sample_comparison_limit),
    )

    summary = {
        "created_at_utc": utc_now(),
        "eval_name": eval_name_final,
        "task_name": task_name,
        "dataset_arrow": str(split_arrow),
        "dataset_jsonl": str(data_jsonl),
        "num_rows": n_rows,
        "teacher_model": teacher_ref,
        "student_model": student_ref,
        "teacher_model_is_local_path": teacher_is_local,
        "student_model_is_local_path": student_is_local,
        "local_models_only": bool(args.local_models_only),
        "teacher_results_json": str(teacher_res) if teacher_res else None,
        "student_results_json": str(student_res) if student_res else None,
        "teacher_samples_jsonl": str(teacher_samples) if teacher_samples else None,
        "student_samples_jsonl": str(student_samples) if student_samples else None,
        "teacher_metric_key": t_metric_key,
        "student_metric_key": s_metric_key,
        "teacher_exact_match": t_score,
        "student_exact_match": s_score,
        "delta_teacher_minus_student": (t_score - s_score) if (t_score is not None and s_score is not None) else None,
        "teacher_output": str(teacher_out),
        "student_output": str(student_out),
        "sample_comparison": sample_cmp,
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_leaderboard(base_root, summary)

    lines = [
        "# Custom lm-eval GSM8K Report",
        "",
        f"- Generated (UTC): `{summary['created_at_utc']}`",
        f"- Task: `{task_name}`",
        f"- Dataset arrow: `{split_arrow}`",
        f"- Dataset jsonl: `{data_jsonl}`",
        f"- Rows: `{n_rows}`",
        "",
        "## Scores",
        "",
        f"- Teacher (`{args.teacher_model}`): `{t_score}` (metric: `{t_metric_key}`)",
        f"- Student (`{args.student_model}`): `{s_score}` (metric: `{s_metric_key}`)",
        f"- Delta (teacher - student): `{summary['delta_teacher_minus_student']}`",
        "",
        "## Sample Head-to-Head (first loaded rows)",
        "",
        f"- Loaded rows: `{sample_cmp['loaded_rows']}`",
        f"- Teacher wins: `{sample_cmp['teacher_wins']}`",
        f"- Student wins: `{sample_cmp['student_wins']}`",
        f"- Both correct: `{sample_cmp['both_correct']}`",
        f"- Both wrong: `{sample_cmp['both_wrong']}`",
        "",
        "## Outputs",
        "",
        f"- Teacher results: `{teacher_res}`",
        f"- Student results: `{student_res}`",
        f"- Teacher samples: `{teacher_samples}`",
        f"- Student samples: `{student_samples}`",
        f"- Comparison: `{out_dir / 'comparison.json'}`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] comparison={out_dir / 'comparison.json'}")
    print(f"[DONE] report={out_dir / 'report.md'}")
    print(f"[DONE] leaderboard={base_root / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
