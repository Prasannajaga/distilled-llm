from __future__ import annotations

import unittest

import torch

from scripts.infer import prepare_chat_input_ids


class _TokenizerChatTemplateString:
    def encode(self, text: str, return_tensors: str | None = None):
        data = [7 + (ord(ch) % 17) for ch in text][:32]
        if return_tensors == "pt":
            return torch.tensor([data], dtype=torch.long)
        return data

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, return_tensors=None):
        rendered = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _TokenizerChatTemplateDictString(_TokenizerChatTemplateString):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, return_tensors=None):
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
        )
        return {"input_ids": rendered}


class _TokenizerChatTemplateTokenStrings(_TokenizerChatTemplateString):
    def convert_tokens_to_ids(self, tokens):
        return [11 + i for i, _ in enumerate(tokens)]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, return_tensors=None):
        return {"input_ids": ["<|user|>", "hello", "<|assistant|>"]}


class TestInferHelpers(unittest.TestCase):
    def test_prepare_chat_input_ids_encodes_rendered_template_strings(self):
        tokenizer = _TokenizerChatTemplateString()
        input_ids = prepare_chat_input_ids(tokenizer, "hello", torch.device("cpu"))
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertEqual(input_ids.dim(), 2)
        self.assertGreater(input_ids.shape[1], 0)

    def test_prepare_chat_input_ids_encodes_string_input_ids_from_dict(self):
        tokenizer = _TokenizerChatTemplateDictString()
        input_ids = prepare_chat_input_ids(tokenizer, "hello", torch.device("cpu"))
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertEqual(input_ids.dim(), 2)
        self.assertGreater(input_ids.shape[1], 0)

    def test_prepare_chat_input_ids_handles_token_string_lists(self):
        tokenizer = _TokenizerChatTemplateTokenStrings()
        input_ids = prepare_chat_input_ids(tokenizer, "hello", torch.device("cpu"))
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertEqual(input_ids.dim(), 2)
        self.assertTrue(torch.equal(input_ids[0], torch.tensor([11, 12, 13], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
