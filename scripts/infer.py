from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATASETS_ROOT = PROJECT_ROOT / "datasets"
if str(DATASETS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASETS_ROOT))

from Cdatasets.tokenizer import load_tokenizer
from scripts.model import GQATransformer
from utils.common import get_device, params
from utils.config import TrainingConfig
from utils.engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved output model")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output model directory (example: ./output/mini-math-student)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint (.pt). Defaults to latest checkpoint in --output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, mps, cpu",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None) 
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional one-shot prompt. If omitted, interactive mode starts.",
    )
    return parser.parse_args()


def find_output_dir(explicit_output: str | None) -> Path:
    if explicit_output:
        output_dir = Path(explicit_output)
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
        return output_dir

    root = Path("output")
    if not root.exists():
        raise FileNotFoundError(
            "No output directory found. Pass --output <dir> with saved artifacts."
        )

    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "config.json").exists()
        and (path / "tokenizer.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No valid model output folders found in ./output. "
            "Each folder must contain config.json and tokenizer.json."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_step(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1


def find_checkpoint(output_dir: Path, explicit_checkpoint: str | None) -> Path:
    if explicit_checkpoint:
        checkpoint = Path(explicit_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoints = sorted(output_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints (*.pt) found in {output_dir}. "
            "Pass --checkpoint explicitly."
        )

    return max(
        checkpoints,
        key=lambda p: (
            _extract_step(p),
            p.stat().st_mtime,
            1 if "final" in p.stem.lower() else 0,
        ),
    )


def load_saved_config(output_dir: Path) -> TrainingConfig:
    cfg = TrainingConfig()
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return cfg

    data: dict[str, Any] = json.loads(config_path.read_text())
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def apply_inference_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_k is not None:
        config.use_top_k = True
        config.top_k = args.top_k
    if args.repetition_penalty is not None:
        config.use_repetition_penalty = True
        config.repetition_penalty = args.repetition_penalty


def build_model(config: TrainingConfig, vocab_size: int) -> GQATransformer:
    return GQATransformer(
        num_layers=config.n_layer,
        n_emb=config.n_embd,
        n_head=config.n_head,
        n_kv_head=max(1, config.n_head // 2),
        vocab_size=vocab_size,
        block_size=config.block_size,
        dropout=0.0,
    )


def prepare_input_ids(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(model_inputs, dict):
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = model_inputs
    except Exception:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]

    if isinstance(input_ids, str):
        input_ids = tokenizer(input_ids, return_tensors="pt")["input_ids"]
    elif not isinstance(input_ids, torch.Tensor):
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        except (TypeError, ValueError):
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    return input_ids.to(device)


def make_human_readable(text: str) -> str:
    out = text

    try:
        from pylatexenc.latex2text import LatexNodes2Text

        out = LatexNodes2Text().latex_to_text(out)
    except Exception:
        # Fallback for common LaTeX markers when pylatexenc is unavailable.
        out = re.sub(r"\\begin\{[^}]+\}", " ", out)
        out = re.sub(r"\\end\{[^}]+\}", " ", out)
        out = re.sub(r"\\[a-zA-Z]+\*?", " ", out)

    out = re.sub(r"\$+", " ", out)
    out = re.sub(r"[_^{}]", " ", out)
    out = re.sub(r"\\+", " ", out)
    out = re.sub(r"\s+", " ", out).strip()

    try:
        from ftfy import fix_text

        out = fix_text(out)
    except Exception:
        pass

    return out


def run_prompt(
    engine: InferenceEngine,
    tokenizer: Any,
    prompt: str,
) -> str:
    input_ids = prepare_input_ids(tokenizer, prompt, engine.device)
    completion = "".join(list(engine.stream_generate(input_ids)))
    return make_human_readable(completion)


def interactive_loop(
    engine: InferenceEngine,
    tokenizer: Any,
) -> None:
    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        print("assistant> ", end="", flush=True)
        completion = run_prompt(engine, tokenizer, prompt)
        print(completion)


def main() -> None:
    args = parse_args()

    output_dir = find_output_dir(args.output)
    checkpoint_path = find_checkpoint(output_dir, args.checkpoint)

    config = load_saved_config(output_dir)
    apply_inference_overrides(config, args)

    requested_device = args.device if args.device != "auto" else "auto"
    device = get_device(requested_device)
    if device.type != "cuda":
        config.use_amp = False

    tokenizer = load_tokenizer(str(output_dir))
    model = build_model(config, vocab_size=len(tokenizer)).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"[MODEL] parameters={params(model)}") 
    model.eval()

    engine = InferenceEngine(model=model, config=config, device=device, tokenizer=tokenizer)

    print(f"[MODEL] output={output_dir}")
    print(f"[CKPT]  {checkpoint_path}")
    print(f"[DEVICE] {device}")
    if args.prompt:
        prompt = " ".join(args.prompt)
        completion = run_prompt(engine, tokenizer, prompt)
        print(completion)
        return

    interactive_loop(engine, tokenizer)


if __name__ == "__main__":
    main()
