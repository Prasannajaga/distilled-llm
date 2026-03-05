from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch


class EvalMetrics:
    """Wrapper for post-train prompt inference and metrics aggregation."""

    def __init__(
        self,
        *,
        prompts: list[str],
        model: Any,
        tokenizer: Any,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        repetition_penalty: float,
        eos_token_id: Optional[int],
    ) -> None:
        self.prompts = list(prompts)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_k = top_k
        self.repetition_penalty = float(repetition_penalty)
        self.eos_token_id = eos_token_id

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _encode_prompt_tensor(self, prompt: str) -> torch.Tensor:
        try:
            encoded = self.tokenizer.encode(prompt, return_tensors="pt")
        except TypeError:
            encoded = self.tokenizer.encode(prompt)
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = torch.tensor([encoded], dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids.to(self.device)

    def _decode_tensor_tokens(self, token_tensor: torch.Tensor) -> str:
        token_ids = token_tensor[0].detach().cpu().tolist()
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(token_ids)

    def run(self, *, global_step: int) -> Dict[str, Any]:
        records: list[Dict[str, Any]] = []
        total_generated_tokens = 0
        total_latency_s = 0.0
        failures = 0

        was_training = self.model.training
        self.model.eval()

        with torch.inference_mode():
            for idx, prompt in enumerate(self.prompts):
                entry: Dict[str, Any] = {
                    "index": int(idx),
                    "prompt": str(prompt),
                }
                try:
                    input_ids = self._encode_prompt_tensor(str(prompt))
                    input_len = int(input_ids.shape[-1])
                    started = time.perf_counter()
                    with torch.amp.autocast(
                        enabled=self.use_amp,
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                    ):
                        output_ids = self.model.generate(
                            input_ids,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_k=self.top_k,
                            repetition_penalty=self.repetition_penalty,
                            eos_token_id=self.eos_token_id,
                        )
                    elapsed_s = max(1e-9, time.perf_counter() - started)
                    output_len = int(output_ids.shape[-1])
                    generated_tokens = max(0, output_len - input_len)
                    completion_text = self._decode_tensor_tokens(output_ids)

                    entry.update(
                        {
                            "success": True,
                            "input_tokens": input_len,
                            "output_tokens": output_len,
                            "generated_tokens": generated_tokens,
                            "latency_ms": round(elapsed_s * 1000.0, 3),
                            "tokens_per_second": round(generated_tokens / elapsed_s, 4),
                            "completion": completion_text,
                        }
                    )
                    total_generated_tokens += generated_tokens
                    total_latency_s += elapsed_s
                except Exception as exc:
                    failures += 1
                    entry.update(
                        {
                            "success": False,
                            "error": str(exc),
                        }
                    )
                records.append(entry)

        if was_training:
            self.model.train()

        successful = len(records) - failures
        avg_latency_ms = (total_latency_s * 1000.0 / successful) if successful > 0 else None
        avg_tokens_per_second = (total_generated_tokens / total_latency_s) if total_latency_s > 0 else None
        return {
            "schema_version": 1,
            "created_at_utc": self.utc_now_iso(),
            "global_step": int(global_step),
            "max_new_tokens": int(self.max_new_tokens),
            "summary": {
                "prompt_count": int(len(records)),
                "success_count": int(successful),
                "failure_count": int(failures),
                "total_generated_tokens": int(total_generated_tokens),
                "avg_latency_ms": round(avg_latency_ms, 3) if avg_latency_ms is not None else None,
                "avg_tokens_per_second": round(avg_tokens_per_second, 4) if avg_tokens_per_second is not None else None,
            },
            "records": records,
        }
