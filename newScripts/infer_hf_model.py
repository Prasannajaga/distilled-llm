from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a Hugging Face causal LM.")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local model directory")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["auto", "float32", "float16", "bfloat16"], default="auto")

    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional input file (.txt, .jsonl). For .jsonl, uses fields: prompt/question/text/input",
    )
    parser.add_argument("--output-file", type=str, default=None, help="Optional output path (.txt or .jsonl)")

    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--render-mode", type=str, choices=["auto", "chat", "plain"], default="auto")

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic sampling")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16

    # auto
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def supports_chat_template(tokenizer: Any) -> bool:
    tpl = getattr(tokenizer, "chat_template", None)
    return bool(tpl)


def build_prompt(tokenizer: Any, user_prompt: str, system_prompt: str, render_mode: str) -> str:
    use_chat = render_mode == "chat" or (render_mode == "auto" and supports_chat_template(tokenizer))
    if use_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {user_prompt}\nAnswer:"


def load_prompts(input_file: str | None, prompt: str | None) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt)

    if not input_file:
        return prompts

    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".txt":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
        return prompts

    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            for key in ("prompt", "question", "text", "input"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    prompts.append(value.strip())
                    break
        return prompts

    raise ValueError("--input-file must be .txt or .jsonl")


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def save_outputs(output_file: str, rows: list[dict[str, Any]]) -> None:
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".jsonl":
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        return

    lines: list[str] = []
    for i, row in enumerate(rows, start=1):
        lines.append(f"[{i}] Prompt: {row['prompt']}")
        lines.append(f"[{i}] Output: {row['output']}")
        lines.append("")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    prompts = load_prompts(args.input_file, args.prompt)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    if prompts:
        rows: list[dict[str, Any]] = []
        for p in prompts:
            rendered = build_prompt(tokenizer, p, args.system_prompt, args.render_mode)
            out = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=rendered,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
            )
            rows.append({"prompt": p, "output": out})

        for i, row in enumerate(rows, start=1):
            print(f"[{i}] Prompt: {row['prompt']}")
            print(f"[{i}] Output: {row['output']}")
            print()

        if args.output_file:
            save_outputs(args.output_file, rows)
            print(f"Saved outputs to: {args.output_file}")

    if not args.no_interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            rendered = build_prompt(tokenizer, user_input, args.system_prompt, args.render_mode)
            out = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=rendered,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
            )
            print(f"Output: {out}\n")


if __name__ == "__main__":
    main()
