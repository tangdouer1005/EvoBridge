"""Paper Section 3.1: exact message-level pattern screening, not redaction."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Released:
    text: str
    blocked: bool


class PolicyGuard:
    BLOCKED = "BLOCKED"

    def __init__(self, prohibited_patterns: Sequence[str]):
        if isinstance(prohibited_patterns, (str, bytes)):
            raise TypeError("Provide a list of regex patterns")
        self.patterns = tuple(prohibited_patterns)
        self._compiled = tuple(re.compile(p) for p in self.patterns)

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.patterns).encode()).hexdigest()

    def release(self, message: str) -> Released:
        if not isinstance(message, str):
            raise TypeError("Only explicit text messages may cross the boundary")
        blocked = any(p.search(message) for p in self._compiled)
        return Released(self.BLOCKED if blocked else message, blocked)
