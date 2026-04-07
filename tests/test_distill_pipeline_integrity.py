from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from newScripts.common import (
    BUCKET_BOTH_CORRECT,
    BUCKET_TEACHER_CORRECT_STUDENT_WRONG,
    BUCKET_TEACHER_WRONG_STUDENT_CORRECT,
    build_phase_paths,
    compute_bucket_metrics,
    compute_overlap_report,
    write_json,
)
from newScripts.run_lm_eval_custom import (
    build_eval_integrity_status,
    build_partial_eval_error_message,
    resolve_limit_config,
)
from newScripts.run_student_distill_from_teacher import _build_recipe_datasets


def _row(idx: int, *, question: str, gold_num: int, teacher_num: int, teacher_ok: bool, student_ok: bool) -> dict:
    return {
        "idx": idx,
        "question": question,
        "gold": f"#### {gold_num}",
        "teacher_pred": str(teacher_num),
        "teacher_raw": f"Reasoning.\nThe answer is {teacher_num}.",
        "teacher_ok": bool(teacher_ok),
        "student_ok": bool(student_ok),
    }


class DistillPipelineIntegrityTests(unittest.TestCase):
    def test_partial_eval_integrity_flags(self) -> None:
        integrity = build_eval_integrity_status(
            expected_rows=2000,
            loaded_rows=1,
            allow_partial_eval=False,
            likely_source="lm-eval --limit=1",
            limit_cfg={"active": True, "value": "1", "source": "cli"},
        )
        self.assertTrue(integrity["partial_eval_detected"])
        msg = build_partial_eval_error_message(integrity)
        self.assertIn("expected_rows=2000", msg)
        self.assertIn("loaded_rows=1", msg)
        self.assertIn("lm-eval --limit=1", msg)

    def test_limit_resolution_ignores_env_by_default(self) -> None:
        env = {"LIMIT": "1", "EVAL_LIMIT": "7"}
        cfg = resolve_limit_config(None, allow_env_limit=False, env=env)
        self.assertFalse(cfg["active"])
        self.assertEqual(cfg["source"], "none")
        self.assertEqual(cfg["ignored_env_limit"], "7")

        cfg_allowed = resolve_limit_config(None, allow_env_limit=True, env=env)
        self.assertTrue(cfg_allowed["active"])
        self.assertEqual(cfg_allowed["value"], "7")
        self.assertEqual(cfg_allowed["source"], "env")

    def test_overlap_report_detects_leakage(self) -> None:
        report = compute_overlap_report(
            {
                "train": [{"question": "What is 2 + 2?", "answer": "4"}],
                "eval": [{"question": "What is 2 + 2?", "answer": "4"}],
            }
        )
        self.assertGreaterEqual(report["max_overlap_pct"], 1.0)
        pair = report["pairwise"][0]
        self.assertEqual(pair["overlap_count"], 1)

    def test_bucket_metrics_synthetic(self) -> None:
        rows = [
            {"teacher_ok": True, "student_ok": False},
            {"teacher_ok": True, "student_ok": True},
            {"teacher_ok": False, "student_ok": False},
            {"teacher_ok": False, "student_ok": True},
        ]
        metrics = compute_bucket_metrics(rows)
        counts = metrics["bucket_counts"]
        self.assertEqual(counts[BUCKET_TEACHER_CORRECT_STUDENT_WRONG], 1)
        self.assertEqual(counts[BUCKET_BOTH_CORRECT], 1)
        self.assertEqual(counts["both wrong"], 1)
        self.assertEqual(counts[BUCKET_TEACHER_WRONG_STUDENT_CORRECT], 1)
        self.assertAlmostEqual(metrics["teacher_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["student_accuracy"], 0.5)

    def test_mixed_bucket_recipe_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase_paths = build_phase_paths(root, "phase-x")
            phase_paths.ensure()
            comparison_json = root / "comparison.json"
            source_rows = []
            idx = 0
            for i in range(10):
                source_rows.append(
                    _row(
                        idx,
                        question=f"tc_sw {i}",
                        gold_num=100 + i,
                        teacher_num=100 + i,
                        teacher_ok=True,
                        student_ok=False,
                    )
                )
                idx += 1
            for i in range(10):
                source_rows.append(
                    _row(
                        idx,
                        question=f"both_correct {i}",
                        gold_num=200 + i,
                        teacher_num=200 + i,
                        teacher_ok=True,
                        student_ok=True,
                    )
                )
                idx += 1
            for i in range(10):
                source_rows.append(
                    _row(
                        idx,
                        question=f"tw_sc {i}",
                        gold_num=300 + i,
                        teacher_num=999 + i,
                        teacher_ok=False,
                        student_ok=True,
                    )
                )
                idx += 1
            write_json(
                comparison_json,
                {
                    "sample_comparison": {
                        "examples": source_rows,
                    }
                },
            )
            dataset_manifest = {
                "comparison_json": str(comparison_json),
                "render_config": {
                    "short_rationale_max_sentences": 3,
                    "short_rationale_max_chars": 360,
                    "max_answer_chars": 2000,
                },
            }
            manifest_path, manifest = _build_recipe_datasets(
                phase_paths=phase_paths,
                dataset_manifest=dataset_manifest,
                winner_format="answer_only",
                seed=42,
            )
            self.assertTrue(manifest_path.exists())
            recipes = manifest["recipes"]
            self.assertEqual(recipes["recipe_pure_target"]["bucket_counts"][BUCKET_TEACHER_CORRECT_STUDENT_WRONG], 10)
            self.assertEqual(recipes["recipe_mix_b"]["bucket_counts"][BUCKET_TEACHER_CORRECT_STUDENT_WRONG], 7)
            self.assertEqual(recipes["recipe_mix_b"]["bucket_counts"][BUCKET_BOTH_CORRECT], 2)
            self.assertEqual(recipes["recipe_mix_b"]["bucket_counts"][BUCKET_TEACHER_WRONG_STUDENT_CORRECT], 1)
            self.assertEqual(recipes["recipe_mix_c"]["bucket_counts"][BUCKET_TEACHER_CORRECT_STUDENT_WRONG], 6)
            self.assertEqual(recipes["recipe_mix_c"]["bucket_counts"][BUCKET_BOTH_CORRECT], 3)
            self.assertEqual(recipes["recipe_mix_c"]["bucket_counts"][BUCKET_TEACHER_WRONG_STUDENT_CORRECT], 1)


if __name__ == "__main__":
    unittest.main()
