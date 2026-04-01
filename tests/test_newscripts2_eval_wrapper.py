from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newScripts2 import eval_wrapper


class EvalWrapperUnitTests(unittest.TestCase):
    def test_resolve_output_dir_never_silent_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            root.mkdir(parents=True, exist_ok=True)
            existing = root / "my-eval"
            existing.mkdir(parents=True, exist_ok=True)
            (existing / "marker.txt").write_text("keep", encoding="utf-8")

            new_path = eval_wrapper.resolve_output_dir(root, "my-eval", overwrite_output=False)
            self.assertNotEqual(new_path, existing)
            self.assertTrue(new_path.exists())
            self.assertTrue((existing / "marker.txt").exists())

    def test_build_lm_eval_command_contains_repro_flags(self) -> None:
        role = eval_wrapper.RoleSpec(
            role="teacher",
            model="/tmp/model",
            model_type="hf",
            model_args_extra=("dtype=float16", "revision=main"),
        )
        cmd = eval_wrapper.build_lm_eval_command(
            python_bin="python",
            role=role,
            task_name="gsm8k_local_test",
            include_path=Path("/tmp/tasks"),
            output_path=Path("/tmp/out"),
            seed=123,
            device="cuda:0",
            batch_size="1",
            num_fewshot=8,
            gen_max_toks=256,
            limit="100",
            local_models_only=True,
            debug=True,
        )

        cmd_str = " ".join(cmd)
        self.assertIn("--seed 123", cmd_str)
        self.assertIn("--num_fewshot 8", cmd_str)
        self.assertIn("--output_path /tmp/out", cmd_str)
        self.assertIn("--limit 100", cmd_str)
        self.assertIn("--verbosity DEBUG", cmd_str)
        self.assertIn("local_files_only=True", cmd_str)

    def test_invalid_dataset_returns_failure_with_summary(self) -> None:
        parser = eval_wrapper.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(
                [
                    "--dataset-path",
                    str(Path(tmp) / "missing_dataset"),
                    "--model",
                    "/tmp/model",
                    "--role",
                    "teacher",
                    "--output-root",
                    str(Path(tmp) / "out"),
                    "--eval-name",
                    "invalid-dataset",
                ]
            )
            rc = eval_wrapper.run_eval(args, cwd=Path(tmp))
            self.assertEqual(rc, 2)

            # A failed run still produces a stable summary for debugging.
            summaries = sorted((Path(tmp) / "out").rglob("summary.json"))
            self.assertTrue(summaries)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error_type"], "wrapper/config error")

    def test_resolve_dataset_source_supports_cached_snapshot_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            requested = tmp_path / "GSM8K"  # Intentionally missing path alias.
            snapshot_dir = tmp_path / "openai___gsm8k" / "main" / "0.0.0" / "abc123"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            arrow_path = snapshot_dir / "gsm8k-test.arrow"
            arrow_path.write_text("placeholder", encoding="utf-8")

            source_kind, source_path = eval_wrapper._resolve_dataset_source(
                str(requested),
                "test",
            )

            self.assertEqual(source_kind, "arrow")
            self.assertEqual(source_path, arrow_path)


class EvalWrapperIntegrationTests(unittest.TestCase):
    def test_mocked_teacher_student_run_writes_standard_artifacts(self) -> None:
        parser = eval_wrapper.build_parser()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.jsonl"
            dataset_rows = [
                {"question": "2+2?", "answer": "#### 4"},
                {"question": "2+3?", "answer": "#### 5"},
            ]
            with dataset_path.open("w", encoding="utf-8") as f:
                for row in dataset_rows:
                    f.write(json.dumps(row) + "\n")

            args = parser.parse_args(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--split",
                    "test",
                    "--max-samples",
                    "2",
                    "--teacher-model",
                    "/tmp/teacher-model",
                    "--student-model",
                    "/tmp/student-model",
                    "--seed",
                    "777",
                    "--eval-name",
                    "integration-run",
                    "--output-root",
                    str(tmp_path / "out"),
                    "--debug",
                    "--comparison-limit",
                    "50",
                ]
            )

            def fake_executor(cmd: list[str], _cwd: Path, log_path: Path) -> None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("fake lm-eval execution\n", encoding="utf-8")

                out_idx = cmd.index("--output_path") + 1
                task_idx = cmd.index("--tasks") + 1
                out_dir = Path(cmd[out_idx])
                task_name = cmd[task_idx]
                out_dir.mkdir(parents=True, exist_ok=True)

                role = "teacher" if "roles/teacher" in str(out_dir) else "student"
                metric = 0.75 if role == "teacher" else 0.35

                results = {"results": {task_name: {"exact_match,marker-priority": metric}}}
                (out_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

                if role == "teacher":
                    samples = [
                        {
                            "doc_id": 0,
                            "doc": {"question": "2+2?", "answer": "#### 4"},
                            "target": "#### 4",
                            "prompt": "Q: 2+2?\\nA:",
                            "arguments": ["Q: 2+2?\\nA:"],
                            "resps": [["4"]],
                            "filtered_resps": ["4"],
                            "exact_match": 1,
                        },
                        {
                            "doc_id": 1,
                            "doc": {"question": "2+3?", "answer": "#### 5"},
                            "target": "#### 5",
                            "prompt": "Q: 2+3?\\nA:",
                            "arguments": ["Q: 2+3?\\nA:"],
                            "resps": [["1"]],
                            "filtered_resps": ["1"],
                            "exact_match": 0,
                        },
                    ]
                else:
                    samples = [
                        {
                            "doc_id": 0,
                            "doc": {"question": "2+2?", "answer": "#### 4"},
                            "target": "#### 4",
                            "prompt": "Q: 2+2?\\nA:",
                            "arguments": ["Q: 2+2?\\nA:"],
                            "resps": [["2"]],
                            "filtered_resps": ["2"],
                            "exact_match": 0,
                        },
                        {
                            "doc_id": 1,
                            "doc": {"question": "2+3?", "answer": "#### 5"},
                            "target": "#### 5",
                            "prompt": "Q: 2+3?\\nA:",
                            "arguments": ["Q: 2+3?\\nA:"],
                            "resps": [["5"]],
                            "filtered_resps": ["5"],
                            "exact_match": 1,
                        },
                    ]

                with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
                    for row in samples:
                        f.write(json.dumps(row) + "\n")

            rc = eval_wrapper.run_eval(args, cwd=tmp_path, command_executor=fake_executor)
            self.assertEqual(rc, 0)

            run_dir = tmp_path / "out" / "integration-run"
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "parsed_metrics.json").exists())
            self.assertTrue((run_dir / "failed_examples.jsonl").exists())
            self.assertTrue((run_dir / "comparison_examples.jsonl").exists())
            self.assertTrue((run_dir / "debug_samples.jsonl").exists())
            self.assertTrue((run_dir / "resolved_config.json").exists())

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 777)
            self.assertEqual(manifest["dataset"]["split"], "test")
            self.assertEqual(len(manifest["roles"]), 2)

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "success")
            buckets = summary["comparison"]["bucket_counts"]
            self.assertEqual(buckets["teacher correct / student wrong"], 1)
            self.assertEqual(buckets["teacher wrong / student correct"], 1)


if __name__ == "__main__":
    unittest.main()
