#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check teacher milestone gate from optimization summary.")
    p.add_argument("--summary-json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.summary_json)
    payload = json.loads(path.read_text(encoding="utf-8"))

    out = {
        "best_teacher_exact_match": payload.get("best_teacher_exact_match"),
        "teacher_em_target": payload.get("teacher_em_target"),
        "target_met": bool(payload.get("target_met", False)),
        "plateau_met": bool(payload.get("plateau_met", False)),
        "improvements_dominate": bool(payload.get("improvements_dominate", False)),
        "distill_gate_passed": bool(payload.get("distill_gate_passed", False)),
    }
    print(json.dumps(out, indent=2))
    if not out["distill_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

