cd /home/prasanna/coding/distilled-llm

RUN_ID=v2
EXP=distill-format-ablation

# Pinned teacher + comparison source (freeze teacher optimization for this phase)
PINNED_SUMMARY="newoutput/teacher_optimization_summary_bootstrap.json"

# Controlled distillation phase:
# 1) Filter teacher-correct/student-wrong
# 2) Build answer-only / short-rationale / full-rationale datasets
# 3) Train and eval 3 identical LoRA runs
# 4) Select winner format
# 5) Run pure-distill vs mixed-gold+distill using winner format
uv run python newScripts/run_student_distill_from_teacher.py \
  --summary-json "$PINNED_SUMMARY" \
  --experiment-name "$EXP" \
  --pipeline-run-id "$RUN_ID" \
  --phase-name "next-distill-phase"

# Build tracking dashboard from all evals
uv run python newScripts/build_eval_tracking_dashboard.py
