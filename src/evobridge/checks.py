"""Trusted local checks are separate from the model's review and official score."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from .harness import Candidate
from .types import Check, CheckResult


@dataclass(frozen=True)
class RegisteredCheck:
    run: Check
    # Mark behavior checks only when they exercise executable input/output.
    behavioral: bool = False


class CheckSuite:
    def __init__(self, checks: Mapping[str, RegisteredCheck] | None = None):
        self.checks = dict(checks or {})

    def run_one(self, name: str, candidate: Candidate) -> CheckResult:
        if name not in self.checks:
            raise ValueError("Unknown local check")
        detached = candidate.copy()
        before = detached.freeze().id
        try:
            result = self.checks[name].run(detached)
            if not isinstance(result, CheckResult) or type(result.passed) is not bool:
                raise TypeError("Checks must return CheckResult with boolean passed")
            if result.name != name:
                raise ValueError("Check result name mismatch")
        except Exception as exc:
            return CheckResult(name, False, f"Check failed: {type(exc).__name__}")
        if detached.freeze().id != before:
            return CheckResult(name, False, "Checks must not modify candidate artifacts")
        return result

    def verify(self, candidate: Candidate) -> tuple[CheckResult, ...]:
        results = [self.run_one(name, candidate) for name in self.checks]
        # Files under skills may be prose or executable operations; require a
        # host-supplied behavioral check for every non-documentation change.
        executable = any(p.startswith("skills/") and PurePosixPath(p).suffix.lower()
                         not in (".md", ".txt", ".rst") for p in candidate.changed_paths)
        if executable and not any(c.behavioral for c in self.checks.values()):
            results.append(CheckResult("executable_behavior", False,
                                       "Executable edits require a registered behavioral check"))
        return tuple(results)
