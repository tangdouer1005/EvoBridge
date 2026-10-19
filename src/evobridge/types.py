"""Benchmark-independent role contracts for Algorithm 1 of EvoBridge."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .harness import Candidate, Harness


class ProtocolError(ValueError):
    """An agent response or environment contract is invalid."""


@dataclass(frozen=True)
class Experience:
    id: str
    task: str
    outcome: Any
    observations: tuple[str, ...] = ()
    feedback: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Experience:
        return cls(**{**value, "observations": tuple(value.get("observations", ()))})


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class CloudAction:
    # Inquiry: ASK, PROPOSE, RETAIN. Clarification additionally: RESPOND, WITHDRAW.
    action: str
    content: str

    def validate(self, allowed: Sequence[str]) -> CloudAction:
        if self.action not in allowed or not isinstance(self.content, str) or not self.content.strip():
            raise ProtocolError("Invalid cloud action or empty content")
        return self


@dataclass(frozen=True)
class Realization:
    status: str  # READY, CLARIFY, STOP
    clarification: str = ""
    reviewed: bool = False
    review: str = ""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str = ""


@dataclass(frozen=True)
class LocalContext:
    harness: Harness
    experience: Experience
    history: tuple[Experience, ...]
    resources: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    round_id: str
    status: str
    reason: str
    before: str
    after: str
    inquiries: int
    clarifications: int
    changed_paths: tuple[str, ...] = ()


class LocalImprover(Protocol):
    model_id: str

    def prepare(self, context: LocalContext) -> str: ...
    def investigate(self, question: str, context: LocalContext, candidate: Candidate | None = None) -> str: ...
    def realize(self, candidate: Candidate, transcript: tuple[Message, ...],
                context: LocalContext, remaining: int) -> Realization: ...


class CloudImprover(Protocol):
    model_id: str

    def decide(self, transcript: tuple[Message, ...], remaining: int) -> CloudAction: ...
    def respond(self, transcript: tuple[Message, ...], remaining_inquiries: int) -> CloudAction: ...


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str


class Worker(Protocol):
    model_id: str

    def execute(self, task: Task, harness: Harness) -> Experience: ...


# Host-defined callbacks must use copies or sandboxed resources. They never
# receive a cloud transport, and benchmark evaluators are intentionally absent.
Check = Callable[[Candidate], CheckResult]
