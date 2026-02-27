from __future__ import annotations

from transformers import AutoTokenizer
from typing import Any
import json
import re
from pathlib import Path
from typing import Final

SPECIAL_TOKENS: Final[list[str]] = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|question|>",
    "<|reasoning|>",
    "<|answer|>",
]

MATH_SYMBOLS: Final[list[str]] = [
    "+", "-", "*", "/", "^", "=", "(", ")", ".", ",", "!", "<", ">",
    "\\frac", "\\sqrt", "\\pi", "\\sum", "\\int", "\\infty",
    "\\times", "\\div", "\\pm", "\\leq", "\\geq", "\\neq",
    "{", "}", "[", "]",
]

DIGITS: Final[list[str]] = [str(d) for d in range(10)]

LETTERS: Final[list[str]] = (
    [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
)

WHITESPACE: Final[list[str]] = [" ", "\n", "\t"]


class MathTokenizer:

    def __init__(self) -> None:
        vocab_list: list[str] = []
        vocab_list.extend(SPECIAL_TOKENS)
        vocab_list.extend(DIGITS)
        vocab_list.extend(MATH_SYMBOLS)
        vocab_list.extend(LETTERS)
        vocab_list.extend(WHITESPACE)

        seen: set[str] = set()
        deduped: list[str] = []
        for tok in vocab_list:
            if tok not in seen:
                seen.add(tok)
                deduped.append(tok)

        self._token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(deduped)}
        self._id_to_token: dict[int, str] = {i: tok for i, tok in enumerate(deduped)}

        self._sorted_tokens: list[str] = sorted(
            [t for t in deduped if t not in set(DIGITS)],
            key=lambda t: len(t),
            reverse=True,
        )

        self._digit_pattern: re.Pattern[str] = re.compile(r"[0-9]")
        self._unk_id: int = self._token_to_id.get("<|pad|>", 0)

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<|pad|>"]

    @property
    def question_id(self) -> int:
        return self._token_to_id["<|question|>"]

    @property
    def reasoning_id(self) -> int:
        return self._token_to_id["<|reasoning|>"]

    @property
    def answer_id(self) -> int:
        return self._token_to_id["<|answer|>"]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isdigit():
                ids.append(self._token_to_id[text[i]])
                i += 1
                continue

            matched = False
            for token in self._sorted_tokens:
                if text[i : i + len(token)] == token:
                    ids.append(self._token_to_id[token])
                    i += len(token)
                    matched = True
                    break

            if not matched:
                ids.append(self._unk_id)
                i += 1

        return ids

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            if token_id in self._id_to_token:
                tokens.append(self._id_to_token[token_id])
            else:
                tokens.append("<|unk|>")
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token_to_id": self._token_to_id,
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> MathTokenizer:
        path = Path(path)
        data = json.loads(path.read_text())
        tokenizer = cls.__new__(cls)
        tokenizer._token_to_id = data["token_to_id"]
        tokenizer._id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tokenizer._sorted_tokens = sorted(
            [t for t in tokenizer._token_to_id if t not in set(DIGITS)],
            key=lambda t: len(t),
            reverse=True,
        )
        tokenizer._digit_pattern = re.compile(r"[0-9]")
        tokenizer._unk_id = tokenizer._token_to_id.get("<|pad|>", 0)
        return tokenizer


class BPEMathTokenizer:
    """Wrapper around a HuggingFace `tokenizers.Tokenizer` (BPE-trained).

    Provides the same interface as `MathTokenizer` so the dataset and
    training scripts can use either tokenizer interchangeably.
    """

    def __init__(self, hf_tokenizer: object) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        if not isinstance(hf_tokenizer, HFTokenizer):
            raise TypeError(
                f"Expected a tokenizers.Tokenizer instance, got {type(hf_tokenizer)}"
            )
        self._tokenizer: HFTokenizer = hf_tokenizer

        required = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
                     "<|question|>", "<|reasoning|>", "<|answer|>"]
        for token in required:
            if self._tokenizer.token_to_id(token) is None:
                raise ValueError(
                    f"Required special token {token!r} not found in vocabulary. "
                    "The tokenizer artifact may be corrupt or trained incorrectly."
                )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id("<|eos|>")

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id("<|pad|>")

    @property
    def unk_id(self) -> int:
        return self._tokenizer.token_to_id("<|unk|>")

    @property
    def question_id(self) -> int:
        return self._tokenizer.token_to_id("<|question|>")

    @property
    def reasoning_id(self) -> int:
        return self._tokenizer.token_to_id("<|reasoning|>")

    @property
    def answer_id(self) -> int:
        return self._tokenizer.token_to_id("<|answer|>")

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (without BOS/EOS post-processing).

        The post-processor adds BOS/EOS automatically via TemplateProcessing,
        but for raw training data we often want bare ids. This method bypasses
        the post-processor by encoding with `add_special_tokens=False`.
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> BPEMathTokenizer:
        """Load a BPE tokenizer from a `tokenizer.json` artifact.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If required special tokens are missing.
        """
        from tokenizers import Tokenizer as HFTokenizer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer artifact not found at {path}. "
                "Run `python -m scripts..train_tokenizer` first."
            )
        hf_tokenizer = HFTokenizer.from_file(str(path))
        return cls(hf_tokenizer)


def load_tokenizer(modelName: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        modelName
    )

    return tokenizer