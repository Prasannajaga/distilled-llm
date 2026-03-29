cd /home/prasanna/coding/distilled-llm

RUN_ID=$(date -u +%Y%m%d-%H%M%S)
EXP=teacher-opt

# 1) Before-distill baseline eval
uv run python newScripts/run_before_distill_eval.py \
  --experiment-name "$EXP" \
  --pipeline-run-id "$RUN_ID" \
  --overwrite-output

BASELINE="newoutput/lm_eval/before-distill-${EXP}-${RUN_ID}/comparison.json"

# 2) Teacher optimization loop (teacher-only)
uv run python newScripts/run_teacher_optimization_loop.py \
  --experiment-name "$EXP" \
  --pipeline-run-id "$RUN_ID" \
  --baseline-comparison-json "$BASELINE"

# 3) Check gate
uv run python newScripts/check_distill_gate.py \
  --summary-json newoutput/teacher_optimization_summary.json

# 4) Student distill (only after gate passes)
uv run python newScripts/run_student_distill_from_teacher.py \
  --summary-json newoutput/teacher_optimization_summary.json

# 5) Build tracking dashboard from all evals
uv run python newScripts/build_eval_tracking_dashboard.py
