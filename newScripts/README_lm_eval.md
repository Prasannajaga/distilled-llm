# Custom lm-eval Setup (Local GSM8K)

This repo now includes a custom `lm-evaluation-harness` pipeline that:

- loads local GSM8K from your mounted dataset cache
- creates a local custom task config (`gsm8k_local_custom`)
- runs **teacher and student sequentially**
- writes comparable outputs and a summary report

## Scripts

- `newScripts/run_lm_eval_custom.sh`
  - simple wrapper to run full eval
- `newScripts/run_lm_eval_custom.py`
  - main runner, task generation, comparison report
- `newScripts/view_lm_eval_results.py`
  - view summary + sample outputs side-by-side

## Run

```bash
./newScripts/run_lm_eval_custom.sh "/media/prasanna/716F26140AED9B67/datasets/GSM8K" 0 test
```

With explicit eval name:

```bash
./newScripts/run_lm_eval_custom.sh "/media/prasanna/716F26140AED9B67/datasets/GSM8K" 0 test "before-distill-v1"
```

Offline/local-model only mode:

```bash
./newScripts/run_lm_eval_custom.sh "/media/prasanna/716F26140AED9B67/datasets/GSM8K" 0 test "before-distill-local" 1
```

Arguments:
- `arg1`: dataset path hint
- `arg2`: max samples (`0` = full split)
- `arg3`: split (`test` or `train`)
- `arg4`: eval name (optional)
- `arg5`: local-models-only (`1` to force local only, no download)

## Local model paths

You can pass local model paths directly to the Python runner:

```bash
uv run python newScripts/run_lm_eval_custom.py \
  --teacher-model /path/to/teacher_model_dir \
  --student-model /path/to/student_model_dir \
  --local-models-only \
  --split test
```

If `--teacher-model` / `--student-model` points to an existing directory, it is used as a local path automatically.

## Outputs

Each run creates:

- `newoutput/lm_eval/custom_gsm8k_<timestamp>/comparison.json`
- `newoutput/lm_eval/custom_gsm8k_<timestamp>/report.md`
- `newoutput/lm_eval/custom_gsm8k_<timestamp>/teacher/...`
- `newoutput/lm_eval/custom_gsm8k_<timestamp>/student/...`

`teacher/` and `student/` include lm-eval native `results.json` and sample logs.

## Leaderboard

Each run also updates:

- `newoutput/lm_eval/leaderboard.jsonl`
- `newoutput/lm_eval/leaderboard.md`
- `newoutput/lm_eval/leaderboard_config.json`

If an eval name already exists, a unique suffix is added automatically (timestamp + index), so runs never overwrite each other.

## View Results

```bash
newScripts/view_lm_eval_results.py newoutput/lm_eval/custom_gsm8k_<timestamp> --limit 20
```
