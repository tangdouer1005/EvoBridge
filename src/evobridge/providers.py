"""Small, explicit OpenAI-compatible Chat Completions transport.

This module uses a configured HTTP endpoint and optional environment credential;
credentials come from the configured environment variable. Requests are never retried. ``tokenizer`` is
an optional text-to-count callable for the deployed model. Without it, UTF-8 byte
length plus message overhead is used as a conservative budgeting estimate, not
an assertion of exact model tokenization or server usage.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import ModelSettings
from .types import Message, ProtocolError


TokenCounter = Callable[[str], int]
Transport = Callable[[Request, float | None], Mapping[str, Any] | bytes | str]


def conservative_token_count(text: str) -> int:
    """Return UTF-8 bytes, a deliberately conservative text token estimate."""
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class UsageReceipt:
    """One attempted completion; absent server token counts remain ``None``."""

    model: str
    success: bool
    attempted: bool = False
    elapsed_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cached_tokens: int | None = None
    completion_reasoning_tokens: int | None = None
    request_id: str | None = None
    error: str | None = None
    estimated_input_tokens: int | None = None
    counter: str = "conservative_utf8_bytes"
    omitted_messages: int = 0


class TokenBudget:
    """Bound tools and compact old messages while retaining system/first/latest input.

    Compaction removes oldest conversational messages and records an explicit
    omission notice. It never invents a summary. Callers should keep their active
    phase protocol in a system message; an oversized mandatory message fails
    closed instead of silently dropping instructions or truncating user input.
    """

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

    def fit(self, messages: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], int]:
        messages = [dict(m) for m in messages]
        input_limit = self.settings.context_tokens - self.settings.max_output_tokens
        if input_limit <= 0:
            raise ProtocolError("Output reservation leaves no input context")
        target = min(input_limit, self.settings.compact_at)
        if self.count_messages(messages) <= target:
            return messages, 0
        required = {i for i, m in enumerate(messages) if m["role"] == "system"}
        first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
        if first_user is not None:
            required.add(first_user)
        if messages:
            required.add(len(messages) - 1)
        removable = [i for i in range(len(messages)) if i not in required]
        removed: set[int] = set()
        for index in removable:
            removed.add(index)
            kept = [m for i, m in enumerate(messages) if i not in removed]
            notice = {"role": "user", "content":
                      f"[Context compaction: {len(removed)} older messages omitted; "
                      "no summary or new evidence was generated.]"}
            # Keep the first user payload first across repeated compactions.
            # It anchors the worker task or local phase, unlike this notice.
            insertion = next((i + 1 for i, m in enumerate(kept) if m["role"] == "user"), len(kept))
            kept.insert(insertion, notice)
            if self.count_messages(kept) <= target:
                return kept, len(removed)
        # A small compact_at may be more restrictive than the physical window.
        # Keep mandatory messages and the omission notice if they still fit.
        if removable and self.count_messages(kept) <= input_limit:
            return kept, len(removed)
        if not removable and self.count_messages(messages) <= input_limit:
            return messages, 0
        raise ProtocolError("Mandatory system/first/latest input exceeds the context budget")


def _http_transport(request: Request, timeout: float | None) -> bytes:
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _decode_response(value: Mapping[str, Any] | bytes | str) -> Mapping[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ProtocolError("Chat completion response must be a JSON object")
    return value


class HTTPChatClient:
    """A single-request chat client with inspectable usage and context accounting.

    ``base_url`` is the API root (for example ``https://host/v1``), or the full
    ``/chat/completions`` URL. ``api_key_env`` names an environment variable,
    never the key itself. The injected ``transport(request, timeout)`` returns
    a JSON mapping, UTF-8 bytes, or JSON text. No request is retried, including
    malformed output. Extra sampling fields are sent explicitly for compatible
    local servers; thinking uses Qwen's ``chat_template_kwargs.enable_thinking``.
    """

    def __init__(self, base_url: str, settings: ModelSettings,
                 api_key_env: str | None = None, timeout: float | None = None,
                 tokenizer: TokenCounter | None = None,
                 transport: Transport | None = None):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Expected an HTTP(S) API URL without credentials/query/fragment")
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.url = (self.base_url if self.base_url.endswith("/chat/completions")
                    else self.base_url + "/chat/completions")
        self.settings = settings
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.budget = TokenBudget(settings, tokenizer)
        self.transport = transport or _http_transport
        self.usage_receipts: list[UsageReceipt] = []

    def complete(self, messages: Sequence[Message | Mapping[str, str]]) -> str:
        started = time.monotonic()
        attempted = False
        response: Mapping[str, Any] = {}
        estimated: int | None = None
        omitted = 0
        try:
            normalized = []
            for message in messages:
                value = ({"role": message.role, "content": message.content}
                         if isinstance(message, Message) else dict(message))
                if (set(value) != {"role", "content"}
                        or value["role"] not in ("system", "user", "assistant")
                        or not isinstance(value["content"], str)):
                    raise ProtocolError("Expected text system/user/assistant messages")
                normalized.append(value)
            if not normalized:
                raise ProtocolError("A completion requires messages")
            fitted, omitted = self.budget.fit(normalized)
            estimated = self.budget.count_messages(fitted)
            payload: dict[str, Any] = {
                "model": self.settings.model, "messages": fitted,
                "max_completion_tokens": self.settings.max_output_tokens,
                "stream": False,
            }
            for name in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                         "repetition_penalty", "reasoning_effort"):
                value = getattr(self.settings, name)
                if value is not None:
                    payload[name] = value
            if self.settings.thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": self.settings.thinking}
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self.api_key_env:
                secret = os.environ.get(self.api_key_env)
                if not secret:
                    raise ValueError("Configured API credential environment variable is empty")
                headers["Authorization"] = "Bearer " + secret
            request = Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              headers=headers, method="POST")
            attempted = True
            response = _decode_response(self.transport(request, self.timeout))
            if response.get("error") is not None:
                raise ProtocolError("Chat endpoint returned an error object")
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ProtocolError("Expected exactly one chat completion choice")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise ProtocolError("Invalid completion choice")
            if choice.get("finish_reason") not in (None, "stop"):
                raise ProtocolError("Completion did not finish normally")
            message = choice.get("message")
            if not isinstance(message, Mapping) or message.get("tool_calls"):
                raise ProtocolError("Expected a text completion without native tool calls")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ProtocolError("Completion content must be nonempty text")
        except Exception as exc:
            if isinstance(exc, HTTPError):
                try:
                    response = _decode_response(exc.read())
                except Exception:
                    pass
                finally:
                    exc.close()
            self._receipt(response, False, type(exc).__name__, estimated, omitted,
                          attempted, time.monotonic() - started)
            raise
        self._receipt(response, True, None, estimated, omitted,
                      attempted, time.monotonic() - started)
        return content

    def _receipt(self, response: Mapping[str, Any], success: bool, error: str | None,
                 estimated: int | None, omitted: int, attempted: bool, elapsed: float) -> None:
        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}

        def known(name: str, source: Mapping[str, Any] = usage) -> int | None:
            value = source.get(name)
            return value if type(value) is int and value >= 0 else None

        prompt_details = usage.get("prompt_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, Mapping) else {}
        completion_details = usage.get("completion_tokens_details")
        completion_details = completion_details if isinstance(completion_details, Mapping) else {}

        self.usage_receipts.append(UsageReceipt(
            model=self.settings.model, success=success, attempted=attempted, elapsed_seconds=elapsed,
            prompt_tokens=known("prompt_tokens"), completion_tokens=known("completion_tokens"),
            total_tokens=known("total_tokens"),
            prompt_cached_tokens=known("cached_tokens", prompt_details),
            completion_reasoning_tokens=known("reasoning_tokens", completion_details),
            request_id=response.get("id") if isinstance(response.get("id"), str) else None,
            error=error, estimated_input_tokens=estimated, counter=self.budget.counter_name,
            omitted_messages=omitted,
        ))
