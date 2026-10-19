"""Application tool-output sizing; model execution is delegated to the runtime."""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence
from .config import ModelSettings
from .types import ProtocolError

TokenCounter = Callable[[str], int]

def conservative_token_count(text: str) -> int:
    """UTF-8 byte upper-bound estimate, not a model tokenizer or billed usage."""
    return len(text.encode("utf-8"))


class TokenBudget:
    """Bound application tool results; the native runtime manages model context."""

    def __init__(self, settings: ModelSettings, tokenizer: TokenCounter | None = None):
        self.settings = settings
        self.tokenizer = tokenizer or conservative_token_count
        self.counter_name = "injected_tokenizer" if tokenizer else "conservative_utf8_bytes"

    def count(self, text: str) -> int:
        count = self.tokenizer(text)
        if type(count) is not int or count < 0:
            raise ValueError("Tokenizer must return a nonnegative integer")
        return count

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        # Framing overhead is an estimate, even with a model-specific text counter.
        return 16 + sum(16 + self.count(m["role"]) + self.count(m["content"])
                        for m in messages)

    def truncate(self, text: str, limit: int) -> str:
        if self.count(text) <= limit:
            return text
        marker = "\n[truncated]"
        if self.count(marker) > limit:
            return ""
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count(text[:middle] + marker) <= limit:
                low = middle
            else:
                high = middle - 1
        return text[:low] + marker

    def tool_result(self, name: str, result: Any) -> str:
        """Produce bounded, valid JSON; large tool data is an explicit preview."""
        text = json.dumps({"tool": name, "result": result}, ensure_ascii=False)
        limit = self.settings.tool_output_tokens
        if self.count(text) <= limit:
            return text
        preview = text
        while True:
            envelope = json.dumps({"tool": name, "truncated": True,
                                   "result_preview": preview}, ensure_ascii=False)
            if self.count(envelope) <= limit:
                return envelope
            if not preview:
                raise ProtocolError("Tool output budget cannot hold a result envelope")
            preview = self.truncate(preview, max(0, self.count(preview)
                                                - (self.count(envelope) - limit) - 16))
