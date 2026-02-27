from utils.config import TrainingConfig
import threading
import time
from typing import Optional, Any

import torch
import torch.nn as nn
from transformers import TextIteratorStreamer

class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.cfg = config
        self.device = device
        self.tokenizer = tokenizer

        self.model.eval()

        self.autocast_dtype = (
            torch.bfloat16
            if self.cfg.amp_dtype == "bfloat16"
            else torch.float16
        )

    def _build_generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": self.cfg.max_new_tokens}
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.use_top_k:
            kwargs["top_k"] = self.cfg.top_k
        if self.cfg.use_repetition_penalty:
            kwargs["repetition_penalty"] = self.cfg.repetition_penalty
        kwargs["do_sample"] = self.cfg.temperature is not None and self.cfg.temperature > 0
        if self.cfg.stop_on_eos and self.tokenizer is not None:
            if self.tokenizer.eos_token_id is not None:
                kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        return kwargs

    def _prepare_chat_inputs(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for chat inference")
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        return {key: value.to(self.device) for key, value in inputs.items()}
 
    # Internal sampling logic (shared) 
    def _sample_next_token(self, logits, idx):
        logits = logits / self.cfg.temperature

        # Repetition penalty (vectorized, cheap)
        if self.cfg.use_repetition_penalty:
            uniq = torch.unique(idx)
            logits[:, uniq] /= self.cfg.repetition_penalty

        # Top-K
        if self.cfg.use_top_k:
            v, _ = torch.topk(logits, self.cfg.top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # Streaming generation (token-by-token)
    @torch.no_grad()
    def stream_generate(self, input_ids):
        """
        input_ids: (1, T) already tokenized
        """
        idx = input_ids.to(self.device)

        for _ in range(self.cfg.max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]

            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
                enabled=self.cfg.use_amp,
            ):
                logits, _ = self.model(idx_cond)

            next_token_logits = logits[:, -1, :]
            next_token = self._sample_next_token(next_token_logits, idx)

            # no need to genreate if it's end of token 
            if (
                self.cfg.stop_on_eos
                and next_token.item() == self.tokenizer.eos_token_id
            ):
                break

            idx = torch.cat([idx, next_token], dim=1)

            yield self.tokenizer.decode(
                next_token[0], skip_special_tokens=True
            )

    # Non-streaming generation
    @torch.no_grad()
    def generate(self, input_ids):
        idx = input_ids.to(self.device)

        for _ in range(self.cfg.max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]

            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
                enabled=self.cfg.use_amp,
            ):
                logits, _ = self.model(idx_cond)

            next_token_logits = logits[:, -1, :]
            next_token = self._sample_next_token(next_token_logits, idx)

            # no need of generation if it reached the end of token 
            if (
                self.cfg.stop_on_eos
                and next_token.item() == self.tokenizer.eos_token_id
            ):
                break

            idx = torch.cat([idx, next_token], dim=1)

        return self.tokenizer.decode(idx[0], skip_special_tokens=False)

    def speculative_decoding(
        self,
        model_inputs: dict[str, torch.Tensor],
        draft_model: nn.Module,
        stream: bool = False,
    ) -> tuple[str, float]:
        """
        Draft-assisted speculative decoding.

        The draft model proposes candidate tokens cheaply; the target model
        verifies them in a single forward pass via HuggingFace's
        `assistant_model` argument.  A background thread drives generation
        so the main thread can consume the TextIteratorStreamer.

        Returns:
            (decoded_text, tokens_per_second)
        """
        if not hasattr(self.model, "generate"):
            raise RuntimeError(
                "speculative_decoding requires a HuggingFace model with a .generate() method."
            )
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for speculative decoding.")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs: dict[str, Any] = {
            **model_inputs,
            **self._build_generation_kwargs(),
            "assistant_model": draft_model,
            "streamer": streamer,
        }

        thread_results: dict[str, Any] = {}

        def _generate() -> None:
            with torch.no_grad():
                thread_results["outputs"] = self.model.generate(**generation_kwargs)

        start_time = time.perf_counter()

        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()

        collected_tokens: list[str] = []
        for token_text in streamer:
            collected_tokens.append(token_text)
            if stream:
                print(token_text, end="", flush=True)

        worker.join()
        elapsed = time.perf_counter() - start_time

        outputs = thread_results["outputs"]
        input_len = model_inputs["input_ids"].shape[-1]
        num_generated = outputs.shape[-1] - input_len
        tokens_per_second = num_generated / elapsed if elapsed > 0 else 0.0

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded, tokens_per_second

    def infer_chat(
        self,
        messages: list[dict],
        streamText: bool = False,
        speculative: bool = False,
        draft_model: Optional[nn.Module] = None,
    ) -> Any: 

        if speculative:
            if draft_model is None:
                raise ValueError(
                    "draft_model must be provided when speculative=True."
                )
            model_inputs = self._prepare_chat_inputs(messages)
            return self.speculative_decoding(model_inputs, draft_model, stream=streamText)

        model_inputs = self._prepare_chat_inputs(messages)
        if streamText:
            return self._stream_chat(model_inputs)
        if hasattr(self.model, "generate"):
            with torch.no_grad():
                output = self.model.generate(**model_inputs, **self._build_generation_kwargs())
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.generate(model_inputs["input_ids"])

    def _stream_chat(self, model_inputs: dict[str, torch.Tensor]):
        if hasattr(self.model, "generate"):
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            kwargs = self._build_generation_kwargs()
            kwargs["streamer"] = streamer
            thread = threading.Thread(
                target=self.model.generate,
                kwargs={**model_inputs, **kwargs},
                daemon=True,
            )
            thread.start()
            try:
                for token in streamer:
                    yield token
            finally:
                thread.join()
        else:
            for token in self.stream_generate(model_inputs["input_ids"]):
                yield token

    @torch.no_grad()
    def load_checkpoint(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model_state_dict"], strict=True)
        if "config" in data and data["config"]:
            from src.config.config import TrainingConfig
            self.cfg = TrainingConfig(**data["config"])
            self.autocast_dtype = (
                torch.bfloat16
                if self.cfg.amp_dtype == "bfloat16"
                else torch.float16
            )
        return data
