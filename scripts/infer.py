from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

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
        "--model-name",
        type=str,
        dest="model_name",
        default=None,
        help="Optional explicit model/checkpoint (.pt) file. Defaults to latest checkpoint in --output.",
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
        checkpoint = Path(str(output_dir) + "/" + explicit_checkpoint)
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
    training_config = data.get("training_config", data)
    if not isinstance(training_config, dict):
        training_config = {}

    for key, value in training_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # Backward-compatible mapping for model-style keys.
    model_config = data.get("model_config", {})
    if isinstance(model_config, dict):
        key_map = {
            "num_hidden_layers": "n_layer",
            "hidden_size": "n_embd",
            "num_attention_heads": "n_head",
            "max_position_embeddings": "block_size",
        }
        for src_key, dst_key in key_map.items():
            if hasattr(cfg, dst_key) and src_key in model_config:
                try:
                    setattr(cfg, dst_key, int(model_config[src_key]))
                except Exception:
                    pass
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
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device)


def format_latex(text: str) -> str:
    out = text
    out = re.sub(r"\\begin\{[^}]+\}", "", out)
    out = re.sub(r"\\end\{[^}]+\}", "", out)
    out = re.sub(r"\\label\{[^}]+\}", "", out)
    out = re.sub(r"\\(?:frac|sqrt)\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", out)
    out = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathbb|operatorname)\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\(?:left|right|Big|big|bigg|Bigg)", "", out)
    out = re.sub(r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|pi|phi|psi|omega)", r"\1", out)
    out = re.sub(r"\\(sin|cos|tan|log|ln|exp|min|max|lim|inf|sup|sum|prod|int)", r"\1", out)
    out = re.sub(r"\\(leq|geq|neq|approx|equiv|sim)", lambda m: {"leq": "<=", "geq": ">=", "neq": "!=", "approx": "≈", "equiv": "≡", "sim": "~"}.get(m.group(1), m.group(0)), out)
    out = re.sub(r"\\(cdot|times|div)", lambda m: {"cdot": "·", "times": "×", "div": "÷"}.get(m.group(1), m.group(0)), out)
    out = re.sub(r"\\(in|notin|subset|supset|cup|cap)", lambda m: {"in": "∈", "notin": "∉", "subset": "⊂", "supset": "⊃", "cup": "∪", "cap": "∩"}.get(m.group(1), m.group(0)), out)
    out = re.sub(r"\\(rightarrow|leftarrow|Rightarrow|Leftarrow|infty)", lambda m: {"rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐", "infty": "∞"}.get(m.group(1), m.group(0)), out)
    out = re.sub(r"\\[a-zA-Z]+\*?", " ", out)
    out = re.sub(r"\$+", "", out)
    out = re.sub(r"([^^])\^\{([^}]*)\}", r"\1^(\2)", out)
    out = re.sub(r"_\{([^}]*)\}", r"_\1", out)
    out = re.sub(r"[{}]", "", out)
    out = re.sub(r"&", " ", out)
    out = re.sub(r"\\{2,}", "\n", out)
    out = re.sub(r"\\\s", " ", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def stream_to_stdout(token_stream: Iterator[str]) -> str:
    chunks: list[str] = []
    for chunk in token_stream:
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()
    raw = "".join(chunks)
    cleaned = format_latex(raw)
    if cleaned != raw:
        print("\n--- Formatted ---")
        print(cleaned)
    return cleaned


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

        input_ids = prepare_input_ids(tokenizer, prompt, engine.device)
        print("assistant> ", end="", flush=True)
        stream_to_stdout(engine.stream_generate(input_ids))


def main() -> None:
    args = parse_args()

    output_dir = find_output_dir(args.output)
    checkpoint_path = find_checkpoint(output_dir, args.model_name)

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
        input_ids = prepare_input_ids(tokenizer, prompt, device)
        stream_to_stdout(engine.stream_generate(input_ids))
        return

    interactive_loop(engine, tokenizer)


if __name__ == "__main__":
    main()
