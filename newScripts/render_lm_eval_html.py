#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render lm-eval run results into Tailwind HTML dashboard")
    p.add_argument("run_dir", type=str, help="Path to lm-eval run dir with comparison.json")
    p.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Root folder containing all eval runs (default: parent of run_dir)",
    )
    p.add_argument("--limit", type=int, default=120, help="Max sample rows shown for selected eval")
    p.add_argument(
        "--per-eval-sample-limit",
        type=int,
        default=0,
        help="Max samples loaded per eval for all-eval sample explorer (<=0 means all)",
    )
    p.add_argument("--title", type=str, default="lm-eval Dashboard", help="HTML title heading")
    p.add_argument("--output", type=str, default=None, help="Output HTML path (default: <run_dir>/report.html)")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and limit > 0 and len(rows) >= limit:
                break
    return rows


def find_samples_file(model_dir: Path) -> Path | None:
    matches = sorted(model_dir.rglob("samples*.jsonl"))
    return matches[0] if matches else None


def extract_text_fields(row: dict[str, Any]) -> tuple[str, str, str, str]:
    doc = row.get("doc", {}) if isinstance(row.get("doc", {}), dict) else {}
    q = doc.get("question", row.get("prompt", ""))
    gold = row.get("target", doc.get("answer", ""))

    raw_pred = ""
    if isinstance(row.get("resps"), list) and row["resps"]:
        first = row["resps"][0]
        if isinstance(first, list) and first:
            raw_pred = first[0]
        else:
            raw_pred = first

    extracted_pred = ""
    if isinstance(row.get("filtered_resps"), list) and row["filtered_resps"]:
        extracted_pred = row["filtered_resps"][0]
    if not extracted_pred:
        extracted_pred = raw_pred

    return str(q), str(gold), str(extracted_pred), str(raw_pred)


def to_float(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def fmt_score(x: Any) -> str:
    v = to_float(x)
    if v is None:
        return "NA"
    return f"{v:.6f}"


def fmt_dt(x: str | None) -> str:
    if not x:
        return "NA"
    try:
        d = datetime.fromisoformat(x.replace("Z", "+00:00"))
        return d.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return x


def clean_answer(x: str) -> str:
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        s = m.group(1).strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def extract_number_like(s: str) -> str | None:
    if not s:
        return None
    nums = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", s)
    if not nums:
        return None
    return nums[-1]


def prediction_matches(pred: str, gold: str) -> bool:
    p = clean_answer(pred)
    g = clean_answer(gold)
    if not g:
        return False
    if p == g:
        return True
    pn = extract_number_like(p)
    gn = extract_number_like(g)
    return pn is not None and gn is not None and pn == gn


def collect_all_evals(
    root_dir: Path,
    per_eval_sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    evals: list[dict[str, Any]] = []
    samples_by_eval: dict[str, list[dict[str, Any]]] = {}

    by_eval_name: dict[str, dict[str, Any]] = {}

    for comp in sorted(root_dir.glob("*/comparison.json")):
        run_dir = comp.parent
        summary = load_json(comp)
        eval_name = str(summary.get("eval_name", run_dir.name))

        summary["eval_name"] = eval_name
        summary["_run_dir"] = str(run_dir)
        summary["_comparison_path"] = str(comp)

        teacher_dir = run_dir / "teacher"
        student_dir = run_dir / "student"
        t_samples = find_samples_file(teacher_dir)
        s_samples = find_samples_file(student_dir)

        merged: list[dict[str, Any]] = []
        teacher_wins = 0
        student_wins = 0
        both_correct = 0
        both_wrong = 0

        if t_samples and s_samples:
            t_rows = read_jsonl(t_samples, limit=per_eval_sample_limit)
            s_rows = read_jsonl(s_samples, limit=per_eval_sample_limit)
            n = min(len(t_rows), len(s_rows))
            for i in range(n):
                tq, tg, tp, traw = extract_text_fields(t_rows[i])
                sq, sg, sp, sraw = extract_text_fields(s_rows[i])
                question = tq or sq
                gold = tg or sg

                t_ok = prediction_matches(tp, gold)
                s_ok = prediction_matches(sp, gold)

                winner = "none"
                if t_ok and not s_ok:
                    winner = "teacher"
                    teacher_wins += 1
                elif s_ok and not t_ok:
                    winner = "student"
                    student_wins += 1
                elif t_ok and s_ok:
                    winner = "both"
                    both_correct += 1
                else:
                    both_wrong += 1

                merged.append(
                    {
                        "idx": i,
                        "question": question,
                        "gold": gold,
                        "teacher": tp,
                        "student": sp,
                        "teacher_raw": traw,
                        "student_raw": sraw,
                        "teacher_ok": t_ok,
                        "student_ok": s_ok,
                        "winner": winner,
                    }
                )

        summary["_samples_count"] = len(merged)
        summary["_teacher_wins"] = teacher_wins
        summary["_student_wins"] = student_wins
        summary["_both_correct"] = both_correct
        summary["_both_wrong"] = both_wrong

        samples_by_eval[eval_name] = merged
        by_eval_name[eval_name] = summary

    lb_path = root_dir / "leaderboard.jsonl"
    if lb_path.exists():
        for row in read_jsonl(lb_path):
            eval_name = str(row.get("eval_name", "")).strip()
            if not eval_name:
                continue
            if eval_name in by_eval_name:
                continue
            row["eval_name"] = eval_name
            row["_run_dir"] = str(root_dir / eval_name)
            row["_comparison_path"] = str(root_dir / eval_name / "comparison.json")
            row["_samples_count"] = len(samples_by_eval.get(eval_name, []))
            row["_teacher_wins"] = 0
            row["_student_wins"] = 0
            row["_both_correct"] = 0
            row["_both_wrong"] = 0
            by_eval_name[eval_name] = row
            samples_by_eval.setdefault(eval_name, [])

    evals = list(by_eval_name.values())
    evals.sort(key=lambda r: str(r.get("created_at_utc", "")), reverse=True)
    return evals, samples_by_eval


def render_html(
    *,
    run_dir: Path,
    current: dict[str, Any],
    evals: list[dict[str, Any]],
    samples_by_eval: dict[str, list[dict[str, Any]]],
    title: str,
    row_limit: int,
) -> str:
    def esc(x: Any) -> str:
        return html.escape(str(x))

    current_eval_name = str(current.get("eval_name", run_dir.name))

    ranked = [e for e in evals if to_float(e.get("teacher_exact_match")) is not None]
    ranked.sort(key=lambda e: float(e.get("teacher_exact_match", 0.0)), reverse=True)

    best_teacher = ranked[0] if ranked else None
    best_student = None
    scored_student = [e for e in evals if to_float(e.get("student_exact_match")) is not None]
    if scored_student:
        best_student = max(scored_student, key=lambda e: float(e.get("student_exact_match", 0.0)))

    scored_count = len(ranked)

    leaderboard_rows: list[str] = []
    for i, e in enumerate(ranked, start=1):
        is_current = str(e.get("eval_name")) == current_eval_name
        row_cls = "bg-indigo-50" if is_current else ""
        current_tag = " <span class='text-xs text-indigo-700'>(current)</span>" if is_current else ""
        leaderboard_rows.append(
            "<tr class='border-b border-slate-200 "
            + row_cls
            + "'>"
            + f"<td class='py-2.5 px-3 text-right font-medium'>{i}</td>"
            + f"<td class='py-2.5 px-3 font-medium'>{esc(e.get('eval_name'))}{current_tag}</td>"
            + f"<td class='py-2.5 px-3'>{esc(fmt_dt(e.get('created_at_utc')))}</td>"
            + f"<td class='py-2.5 px-3 text-right'>{esc(e.get('num_rows'))}</td>"
            + f"<td class='py-2.5 px-3 text-right'>{esc(fmt_score(e.get('teacher_exact_match')))}</td>"
            + f"<td class='py-2.5 px-3 text-right'>{esc(fmt_score(e.get('student_exact_match')))}</td>"
            + f"<td class='py-2.5 px-3 text-right'>{esc(fmt_score(e.get('delta_teacher_minus_student')))}</td>"
            + "</tr>"
        )

    cards: list[str] = []
    for e in evals:
        em_t = fmt_score(e.get("teacher_exact_match"))
        em_s = fmt_score(e.get("student_exact_match"))
        delta = fmt_score(e.get("delta_teacher_minus_student"))
        is_current = str(e.get("eval_name")) == current_eval_name

        ring = "ring-2 ring-indigo-300" if is_current else ""

        cards.append(
            "<button type='button' "
            + f"data-eval='{esc(e.get('eval_name'))}' "
            + "class='eval-card text-left rounded-2xl border border-slate-200 bg-white p-4 shadow-sm hover:shadow transition "
            + ring
            + "'>"
            + f"<div class='text-xs text-slate-500'>{esc(fmt_dt(e.get('created_at_utc')))}</div>"
            + f"<div class='text-base font-semibold text-slate-900 mt-1 truncate'>{esc(e.get('eval_name'))}</div>"
            + f"<div class='mt-2 text-sm text-slate-700'>Rows: <b>{esc(e.get('num_rows'))}</b> | Samples: <b>{esc(e.get('_samples_count'))}</b></div>"
            + f"<div class='mt-1 text-sm text-slate-700'>Teacher EM: <b>{esc(em_t)}</b></div>"
            + f"<div class='mt-1 text-sm text-slate-700'>Student EM: <b>{esc(em_s)}</b></div>"
            + f"<div class='mt-1 text-sm text-slate-700'>Delta: <b>{esc(delta)}</b></div>"
            + f"<div class='mt-2 text-xs text-slate-500'>H2H (loaded samples): T {esc(e.get('_teacher_wins'))} / S {esc(e.get('_student_wins'))} / Both {esc(e.get('_both_correct'))}</div>"
            + "</button>"
        )

    sample_data = json.dumps(samples_by_eval, ensure_ascii=False).replace("</", "<\\/")
    eval_options = "".join(
        f"<option value='{esc(e.get('eval_name'))}' {'selected' if str(e.get('eval_name')) == current_eval_name else ''}>{esc(e.get('eval_name'))}</option>"
        for e in evals
    )

    best_teacher_text = (
        f"{fmt_score(best_teacher.get('teacher_exact_match'))} ({best_teacher.get('eval_name')})" if best_teacher else "NA"
    )
    best_student_text = (
        f"{fmt_score(best_student.get('student_exact_match'))} ({best_student.get('eval_name')})" if best_student else "NA"
    )

    teacher_model = esc(current.get("teacher_model", "NA"))
    student_model = esc(current.get("student_model", "NA"))

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{esc(title)} - {esc(current_eval_name)}</title>
  <script src=\"https://cdn.tailwindcss.com\"></script>
</head>
<body class=\"min-h-screen bg-gradient-to-b from-slate-100 to-slate-50 text-slate-900\">
  <main class=\"max-w-7xl mx-auto px-4 py-8 space-y-8\">
    <section class=\"rounded-3xl bg-white/95 border border-slate-200 shadow-sm p-6\">
      <div class=\"flex flex-wrap items-start justify-between gap-4\">
        <div>
          <h1 class=\"text-2xl md:text-3xl font-bold tracking-tight\">{esc(title)}</h1>
          <p class=\"text-slate-600 mt-1\">Current eval: <span class=\"font-medium\">{esc(current_eval_name)}</span></p>
          <p class=\"text-xs text-slate-500 mt-1\">Task: {esc(current.get('task_name', 'NA'))} | Rows: {esc(current.get('num_rows', 'NA'))} | Created: {esc(fmt_dt(current.get('created_at_utc')))}</p>
        </div>
      </div>

      <div class=\"grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-5\">
        <div class=\"rounded-xl border border-slate-200 bg-slate-50 p-4\"><div class=\"text-xs text-slate-500\">Teacher EM (current)</div><div class=\"text-2xl font-semibold mt-1\">{esc(fmt_score(current.get('teacher_exact_match')))}</div></div>
        <div class=\"rounded-xl border border-slate-200 bg-slate-50 p-4\"><div class=\"text-xs text-slate-500\">Student EM (current)</div><div class=\"text-2xl font-semibold mt-1\">{esc(fmt_score(current.get('student_exact_match')))}</div></div>
        <div class=\"rounded-xl border border-slate-200 bg-slate-50 p-4\"><div class=\"text-xs text-slate-500\">Delta (T-S)</div><div class=\"text-2xl font-semibold mt-1\">{esc(fmt_score(current.get('delta_teacher_minus_student')))}</div></div>
        <div class=\"rounded-xl border border-slate-200 bg-slate-50 p-4\"><div class=\"text-xs text-slate-500\">All evals</div><div class=\"text-2xl font-semibold mt-1\">{len(evals)}</div><div class=\"text-xs text-slate-500 mt-1\">scored: {scored_count}</div></div>
      </div>

      <div class=\"grid sm:grid-cols-2 gap-4 mt-4\">
        <div class=\"rounded-xl border border-slate-200 bg-slate-50 p-4\"><div class=\"text-xs text-slate-500\">Best Teacher EM</div><div class=\"text-sm font-medium mt-1 break-all\">{esc(best_teacher_text)}</div></div>
        <div class=\"rounded-xl border border-slate-200 bg-slate-50 p-4\"><div class=\"text-xs text-slate-500\">Best Student EM</div><div class=\"text-sm font-medium mt-1 break-all\">{esc(best_student_text)}</div></div>
      </div>

      <div class=\"mt-4 text-xs text-slate-500 break-all\">Teacher model: {teacher_model}</div>
      <div class=\"mt-1 text-xs text-slate-500 break-all\">Student model: {student_model}</div>
    </section>

    <section class=\"rounded-3xl bg-white/95 border border-slate-200 shadow-sm p-6\">
      <div class=\"flex items-center justify-between gap-3\">
        <div>
          <h2 class=\"text-xl font-semibold\">Leaderboard (All Evals)</h2>
          <p class=\"text-sm text-slate-600 mt-1\">Sorted by Teacher EM descending</p>
        </div>
      </div>
      <div class=\"overflow-x-auto mt-4\">
        <table class=\"w-full text-sm\">
          <thead>
            <tr class=\"border-b border-slate-200 bg-slate-50\">
              <th class=\"py-2.5 px-3 text-right\">Rank</th>
              <th class=\"py-2.5 px-3 text-left\">Eval</th>
              <th class=\"py-2.5 px-3 text-left\">Created</th>
              <th class=\"py-2.5 px-3 text-right\">Rows</th>
              <th class=\"py-2.5 px-3 text-right\">Teacher EM</th>
              <th class=\"py-2.5 px-3 text-right\">Student EM</th>
              <th class=\"py-2.5 px-3 text-right\">Delta</th>
            </tr>
          </thead>
          <tbody>{''.join(leaderboard_rows) if leaderboard_rows else "<tr><td class='py-3 px-3 text-slate-500' colspan='7'>No scored evals yet.</td></tr>"}</tbody>
        </table>
      </div>
    </section>

    <section class=\"rounded-3xl bg-white/95 border border-slate-200 shadow-sm p-6\">
      <h2 class=\"text-xl font-semibold\">All Eval Comparisons</h2>
      <p class=\"text-sm text-slate-600 mt-1\">Click any card to switch sample explorer to that eval.</p>
      <div class=\"grid md:grid-cols-2 xl:grid-cols-3 gap-4 mt-4\">
        {''.join(cards) if cards else "<div class='text-slate-500 text-sm'>No eval runs found.</div>"}
      </div>
    </section>

    <section class=\"rounded-3xl bg-white/95 border border-slate-200 shadow-sm p-6\">
      <div class=\"flex flex-wrap gap-3 items-end\">
        <div>
          <label class=\"block text-xs text-slate-500 mb-1\">Eval</label>
          <select id=\"evalSelect\" class=\"border border-slate-300 rounded-lg px-3 py-2 text-sm bg-white\">{eval_options}</select>
        </div>
        <div class=\"flex-1 min-w-[220px]\">
          <label class=\"block text-xs text-slate-500 mb-1\">Search question/prediction</label>
          <input id=\"searchInput\" class=\"w-full border border-slate-300 rounded-lg px-3 py-2 text-sm\" placeholder=\"Type to filter...\" />
        </div>
        <label class=\"inline-flex items-center gap-2 text-sm text-slate-700\"><input id=\"diffOnly\" type=\"checkbox\" class=\"rounded\" /> Show only disagreements</label>
        <label class=\"inline-flex items-center gap-2 text-sm text-slate-700\"><input id=\"matchOnly\" type=\"checkbox\" class=\"rounded\" /> Show only exact matches</label>
      </div>

      <div class=\"overflow-x-auto mt-4\">
        <table class=\"w-full text-sm\">
          <thead>
            <tr class=\"border-b border-slate-200 bg-slate-50\">
              <th class=\"py-2.5 px-3 text-left\">#</th>
              <th class=\"py-2.5 px-3 text-left\">Winner</th>
              <th class=\"py-2.5 px-3 text-left\">Question</th>
              <th class=\"py-2.5 px-3 text-left\">Gold</th>
              <th class=\"py-2.5 px-3 text-left\">Teacher ✓</th>
              <th class=\"py-2.5 px-3 text-left\">Teacher Extracted</th>
              <th class=\"py-2.5 px-3 text-left\">Teacher Raw</th>
              <th class=\"py-2.5 px-3 text-left\">Student ✓</th>
              <th class=\"py-2.5 px-3 text-left\">Student Extracted</th>
              <th class=\"py-2.5 px-3 text-left\">Student Raw</th>
            </tr>
          </thead>
          <tbody id=\"samplesBody\"></tbody>
        </table>
      </div>
      <p id=\"samplesInfo\" class=\"mt-3 text-xs text-slate-500\"></p>
    </section>
  </main>

  <script>
    const rowLimit = {max(1, int(row_limit))};
    const samplesByEval = {sample_data};
    const evalSelect = document.getElementById('evalSelect');
    const searchInput = document.getElementById('searchInput');
    const diffOnly = document.getElementById('diffOnly');
    const matchOnly = document.getElementById('matchOnly');
    const samplesBody = document.getElementById('samplesBody');
    const samplesInfo = document.getElementById('samplesInfo');

    function esc(s) {{
      return String(s)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }}

    function winnerBadge(w) {{
      if (w === 'teacher') return '<span class="inline-flex rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 text-xs font-medium">Teacher</span>';
      if (w === 'student') return '<span class="inline-flex rounded-full bg-sky-100 text-sky-800 px-2 py-0.5 text-xs font-medium">Student</span>';
      if (w === 'both') return '<span class="inline-flex rounded-full bg-violet-100 text-violet-800 px-2 py-0.5 text-xs font-medium">Both</span>';
      return '<span class="inline-flex rounded-full bg-slate-100 text-slate-700 px-2 py-0.5 text-xs font-medium">None</span>';
    }}

    function matchBadge(ok) {{
      return ok
        ? '<span class="inline-flex rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 text-xs font-semibold">✅</span>'
        : '<span class="inline-flex rounded-full bg-rose-100 text-rose-800 px-2 py-0.5 text-xs font-semibold">❌</span>';
    }}

    function renderSamples() {{
      const evalName = evalSelect.value;
      const q = searchInput.value.trim().toLowerCase();
      const allRows = samplesByEval[evalName] || [];
      const filtered = allRows.filter(r => {{
        if (diffOnly.checked && String(r.teacher).trim() === String(r.student).trim()) return false;
        if (matchOnly.checked && !(r.teacher_ok || r.student_ok)) return false;
        if (!q) return true;
        const blob = `${{r.question}} ${{r.gold}} ${{r.teacher}} ${{r.student}} ${{r.teacher_raw || ''}} ${{r.student_raw || ''}} ${{r.winner}}`.toLowerCase();
        return blob.includes(q);
      }});

      const rows = filtered.slice(0, rowLimit);
      if (!rows.length) {{
        samplesBody.innerHTML = '<tr><td class="py-3 px-3 text-slate-500" colspan="10">No samples found for this filter.</td></tr>';
        samplesInfo.textContent = `Eval: ${{evalName}} | 0 / ${{allRows.length}} rows shown`;
        return;
      }}

      samplesBody.innerHTML = rows.map(r => `
        <tr class="border-b border-slate-200 align-top">
          <td class="py-2.5 px-3 text-slate-500">${{esc(r.idx)}}</td>
          <td class="py-2.5 px-3">${{winnerBadge(r.winner)}}</td>
          <td class="py-2.5 px-3">${{esc(r.question)}}</td>
          <td class="py-2.5 px-3">${{esc(r.gold)}}</td>
          <td class="py-2.5 px-3">${{matchBadge(!!r.teacher_ok)}}</td>
          <td class="py-2.5 px-3">${{esc(r.teacher)}}</td>
          <td class="py-2.5 px-3">${{esc(r.teacher_raw || '')}}</td>
          <td class="py-2.5 px-3">${{matchBadge(!!r.student_ok)}}</td>
          <td class="py-2.5 px-3">${{esc(r.student)}}</td>
          <td class="py-2.5 px-3">${{esc(r.student_raw || '')}}</td>
        </tr>
      `).join('');
      samplesInfo.textContent = `Eval: ${{evalName}} | ${{rows.length}} / ${{allRows.length}} rows shown`;
    }}

    evalSelect.addEventListener('change', renderSamples);
    searchInput.addEventListener('input', renderSamples);
    diffOnly.addEventListener('change', renderSamples);
    matchOnly.addEventListener('change', renderSamples);

    document.querySelectorAll('.eval-card').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const evalName = btn.getAttribute('data-eval');
        if (!evalName) return;
        evalSelect.value = evalName;
        renderSamples();
        window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
      }});
    }});

    renderSamples();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    comp = run_dir / "comparison.json"
    if not comp.exists():
        raise FileNotFoundError(f"comparison.json not found in {run_dir}")

    current = load_json(comp)
    root_dir = Path(args.root_dir) if args.root_dir else run_dir.parent
    evals, samples_by_eval = collect_all_evals(root_dir, per_eval_sample_limit=int(args.per_eval_sample_limit))

    html_text = render_html(
        run_dir=run_dir,
        current=current,
        evals=evals,
        samples_by_eval=samples_by_eval,
        title=args.title,
        row_limit=max(1, int(args.limit)),
    )
    out = Path(args.output) if args.output else (run_dir / "report.html")
    out.write_text(html_text, encoding="utf-8")
    print(f"[DONE] html={out}")


if __name__ == "__main__":
    main()
