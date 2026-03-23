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
    parser.add_argument("--stop_on_eos", type=int, choices=[0, 1], default=None)
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
        candidate = Path(explicit_checkpoint).expanduser()
        checkpoint = candidate if candidate.is_absolute() else (output_dir / candidate)
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
    if args.stop_on_eos is not None:
        config.stop_on_eos = bool(args.stop_on_eos)
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


def _as_batched_input_ids(inputs: Any, device: torch.device, tokenizer: Any | None = None) -> torch.Tensor:
    def _encode_text(text: str) -> Any:
        if tokenizer is None:
            raise TypeError("String input payload requires a tokenizer for encoding.")
        try:
            return tokenizer.encode(text, return_tensors="pt")
        except TypeError:
            return tokenizer.encode(text)

    if isinstance(inputs, dict):
        if "input_ids" in inputs:
            inputs = inputs["input_ids"]
        elif "text" in inputs:
            inputs = _encode_text(str(inputs["text"]))
        elif "prompt" in inputs:
            inputs = _encode_text(str(inputs["prompt"]))
    if isinstance(inputs, str):
        inputs = _encode_text(inputs)

    if not isinstance(inputs, torch.Tensor) and hasattr(inputs, "input_ids"):
        inputs = getattr(inputs, "input_ids")

    if isinstance(inputs, (list, tuple)) and inputs:
        first = inputs[0]
        if isinstance(first, str):
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
                token_ids = tokenizer.convert_tokens_to_ids(list(inputs))
                if isinstance(token_ids, list) and token_ids and all(isinstance(t, int) for t in token_ids):
                    inputs = token_ids
                else:
                    inputs = _encode_text(" ".join(str(t) for t in inputs))
            else:
                inputs = _encode_text(" ".join(str(t) for t in inputs))
        elif isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
                converted_rows = []
                ok = True
                for row in inputs:
                    row_ids = tokenizer.convert_tokens_to_ids(list(row))
                    if not isinstance(row_ids, list) or not all(isinstance(t, int) for t in row_ids):
                        ok = False
                        break
                    converted_rows.append(row_ids)
                inputs = converted_rows if ok else [_encode_text(" ".join(str(t) for t in row)) for row in inputs]
            else:
                inputs = [_encode_text(" ".join(str(t) for t in row)) for row in inputs]

    if not isinstance(inputs, torch.Tensor):
        try:
            inputs = torch.tensor([inputs], dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Unsupported tokenizer output type for input_ids conversion: {type(inputs)!r}"
            ) from exc
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    return inputs.to(device)


def prepare_input_ids(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
    except TypeError:
        input_ids = tokenizer.encode(prompt)
    return _as_batched_input_ids(input_ids, device, tokenizer=tokenizer)


def prepare_chat_input_ids(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prepare_input_ids(tokenizer, prompt, device)

    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        return prepare_input_ids(tokenizer, prompt, device)

    return _as_batched_input_ids(inputs, device, tokenizer=tokenizer)


def load_state_dict_with_tril_fallback(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        message = str(exc)
        if ".attn.tril" not in message or "size mismatch" not in message:
            raise

    # Allow block_size changes by skipping cached causal-mask buffers only.
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".attn.tril")}
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.endswith(".attn.tril")]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load fallback failed. "
            f"missing_non_tril={missing}, unexpected={unexpected}"
        )
    print("[CKPT] Loaded with tril-buffer fallback (block_size differs from checkpoint).")


def stream_to_stdout(token_stream: Iterator[str]) -> str:
    chunks: list[str] = []
    for chunk in token_stream:
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()
    return "".join(chunks)


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

        input_ids = prepare_chat_input_ids(tokenizer, prompt, engine.device)
        print("assistant> ", end="", flush=True)
        text = stream_to_stdout(engine.stream_generate(input_ids))
        if not text.strip():
            print("[WARN] Empty generation (likely hit EOS immediately). Try a different prompt or higher temperature.")


def main() -> None:
    args = parse_args()

    output_dir = find_output_dir(args.output)
    checkpoint_path = find_checkpoint(output_dir, args.model_name)

    config = load_saved_config(output_dir)
    apply_inference_overrides(config, args)

    device = get_device(args.device)
    if device.type != "cuda":
        config.use_amp = False

    tokenizer = load_tokenizer(str(output_dir))
    model = build_model(config, vocab_size=len(tokenizer)).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    load_state_dict_with_tril_fallback(model, state_dict)
    print(f"[MODEL] parameters={params(model)}") 
    model.eval()

    engine = InferenceEngine(model=model, config=config, device=device, tokenizer=tokenizer)

    print(f"[MODEL] output={output_dir}")
    print(f"[CKPT]  {checkpoint_path}")
    print(f"[DEVICE] {device}")
    if args.prompt:
        prompt = " ".join(args.prompt)
        input_ids = prepare_chat_input_ids(tokenizer, prompt, device)
        text = stream_to_stdout(engine.stream_generate(input_ids))
        if not text.strip():
            print("[WARN] Empty generation (likely hit EOS immediately). Try --temperature 0.9 --top_k 100.")
        return

    interactive_loop(engine, tokenizer)


if __name__ == "__main__":
    main()
