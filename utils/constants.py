"""Fixed identifiers and non-configurable constants for the tokenizer subsystem."""

from __future__ import annotations

from typing import Final

SPECIAL_TOKENS: Final[list[str]] = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|question|>",
    "<|reasoning|>",
    "<|answer|>",
]

MATH_LATEX_TOKENS: Final[list[str]] = [
    "\\frac", "\\sqrt", "\\pi", "\\sum", "\\int", "\\infty",
    "\\times", "\\div", "\\pm", "\\leq", "\\geq", "\\neq",
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta",
    "\\lambda", "\\mu", "\\sigma", "\\phi", "\\omega",
    "\\partial", "\\nabla", "\\forall", "\\exists",
    "\\rightarrow", "\\leftarrow", "\\Rightarrow", "\\Leftarrow",
    "\\subset", "\\supset", "\\subseteq", "\\supseteq",
    "\\cup", "\\cap", "\\in", "\\notin",
    "\\lim", "\\log", "\\ln", "\\sin", "\\cos", "\\tan",
    "\\exp", "\\max", "\\min", "\\sup", "\\inf",
    "\\binom", "\\choose", "\\cdot", "\\ldots", "\\cdots",
    "\\text", "\\mathrm", "\\mathbb", "\\mathcal",
]

TOKENIZER_ARTIFACT_FILENAME: Final[str] = "tokenizer.json"
TOKENIZER_CONFIG_FILENAME: Final[str] = "tokenizer_config.json"

# AutoMathText dataset column identifiers
AUTOMATHTEXT_COL_TEXT: Final[str] = "text"
AUTOMATHTEXT_COL_META: Final[str] = "meta"
AUTOMATHTEXT_COL_TITLE: Final[str] = "title"
AUTOMATHTEXT_COL_URL: Final[str] = "url"
AUTOMATHTEXT_COL_ABSTRACT: Final[str] = "abstract"
