"""Frozen runtime configuration; Appendix B.3 values are explicit profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math


@dataclass(frozen=True)
class ModelSettings:
    model: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None
    context_tokens: int = 131072
    compact_at: int = 98304
    max_output_tokens: int = 32768
    tool_output_tokens: int = 8192

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("A nonempty model name is required")
        for n in (self.context_tokens, self.compact_at, self.max_output_tokens, self.tool_output_tokens):
            if type(n) is not int or n <= 0:
                raise ValueError("Token limits must be positive integers")
        if not 0 < self.compact_at < self.context_tokens:
            raise ValueError("Invalid model/context configuration")
        if self.thinking is not None and type(self.thinking) is not bool:
            raise ValueError("thinking must be boolean or omitted")
        for name in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"Invalid numeric setting: {name}")

    @classmethod
    def local(cls, size: str = "9b") -> ModelSettings:
        if size not in ("9b", "27b"):
            raise ValueError("Expected 9b or 27b")
        return cls(model="Qwen3.5-9B" if size == "9b" else "Qwen3.8-27B",
                   temperature=0.6 if size == "9b" else 1.0, top_p=0.95,
                   top_k=20, min_p=0, presence_penalty=0,
                   repetition_penalty=1.0, thinking=False)


@dataclass(frozen=True)
class RunConfig:
    local: ModelSettings = field(default_factory=ModelSettings.local)
    cloud: ModelSettings = field(default_factory=lambda: ModelSettings(
        model="gpt6-sol", reasoning_effort="medium"))
    inquiry_budget: int = 3
    clarification_budget: int = 3

    def __post_init__(self):
        for n in (self.inquiry_budget, self.clarification_budget):
            if type(n) is not int or n < 0:
                raise ValueError("Interaction budgets must be nonnegative integers")

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict) -> RunConfig:
        return cls(**{**value, "local": ModelSettings(**value["local"]),
                      "cloud": ModelSettings(**value["cloud"])})
