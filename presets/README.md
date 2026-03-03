# Deploy presets

`deploy.py` now accepts `@args-file` syntax via argparse.

## Run with a preset

```bash
uv run python deploy.py @presets/deploy_l4x2_baseline.args
uv run python deploy.py @presets/deploy_l4x2_balanced.args
uv run python deploy.py @presets/deploy_l4x2_longctx.args
```

## Override only what you need

CLI flags provided after `@file` override file values.

```bash
uv run python deploy.py @presets/deploy_l4x2_baseline.args \
  --project_id my-project \
  --region us-central1 \
  --bucket_uri gs://my-bucket \
  --display_name test-run-01
```

## Notes

- Replace `YOUR_PROJECT_ID` and `gs://YOUR_BUCKET` in the preset file (or override via CLI).
- `datasets_json` is set to empty string in presets, so default dataset mixture in `scripts/loadDataset.py` is used.
