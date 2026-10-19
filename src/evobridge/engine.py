"""One evolution round, following final-paper Algorithm 1.

Only explicitly released text is passed to the cloud role. Local checks are
publication prerequisites, never comparisons against a held-out task score.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Mapping

from .checks import CheckSuite
from .config import RunConfig
from .harness import Candidate, Harness
from .policy import PolicyGuard
from .storage import SnapshotStore
from .types import (CloudImprover, Experience, LocalContext, LocalImprover,
                    Message, ProtocolError, Realization, RoundResult)


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


class PendingRound(RuntimeError):
    """An interrupted round cannot be silently repeated."""


class Engine:
    def __init__(self, store: SnapshotStore, local: LocalImprover,
                 cloud: CloudImprover, guard: PolicyGuard,
                 checks: CheckSuite | None = None, config: RunConfig | None = None,
                 *, initial: Harness | None = None,
                 resources: Mapping[str, str] | None = None):
        self.store, self.local, self.cloud, self.guard = store, local, cloud, guard
        self.config = config or RunConfig()
        self.checks = checks or CheckSuite()
        self.resources = dict(resources or {})
        if local.model_id != self.config.local.model or cloud.model_id != self.config.cloud.model:
            raise ValueError("Agent models must match the frozen run configuration")
        for agent, expected in ((local, self.config.local), (cloud, self.config.cloud)):
            settings = getattr(getattr(agent, "client", None), "settings", None)
            if settings is not None and settings != expected:
                raise ValueError("Agent inference settings differ from the frozen configuration")
        self.identity = identity({"run": asdict(self.config), "policy": guard.identity,
                                  "resources": self.resources,
                                  "checks": {k: v.behavioral for k, v in self.checks.checks.items()}})
        self.store.initialize(initial or Harness({"memory.md": "", "CONTEXT.md": ""}),
                              self.identity)

    def evolve(self, experience: Experience, *, round_id: str | None = None,
               history: tuple[Experience, ...] | None = None) -> RoundResult:
        """Adapt only from completed authorized experience, once per round id.

        A completed call is idempotent. An interrupted start is left pending for
        inspection rather than repeating a potentially billed model request.
        Offline schedules may revisit an experience with distinct epoch ids.
        """
        round_id = round_id or experience.id
        input_id = identity({"experience": experience.to_dict(),
                             "history": ([e.to_dict() for e in history] if history is not None
                                         else "stored_experience_at_first_execution")})
        previous = self.store.read_record("rounds", round_id)
        if previous is not None:
            if previous["input_id"] != input_id:
                raise ValueError("Completed round id reused with different experience")
            return RoundResult(**{**previous["result"],
                                  "changed_paths": tuple(previous["result"]["changed_paths"])})
        if self.store.read_record("round-starts", round_id) is not None:
            raise PendingRound("Interrupted round requires inspection; no calls were repeated")
        base = self.store.current()
        claimed = self.store.save_record("round-starts", round_id,
                                         {"input_id": input_id, "before": base.id})
        if not claimed:
            raise PendingRound("Another process already claimed this round")
        self.store.save_record("experience", experience.id, experience.to_dict())
        if history is None:
            history = tuple(Experience.from_dict(r) for r in self.store.history("experience"))
        # Model adapters receive detached local payloads: accidental mutation of
        # a nested outcome must not alter the caller's fixed offline corpus.
        context = LocalContext(base, Experience.from_dict(json.loads(json.dumps(experience.to_dict()))),
                               tuple(Experience.from_dict(json.loads(json.dumps(e.to_dict()))) for e in history),
                               dict(self.resources))
        transcript: list[Message] = []
        inquiries = clarifications = 0
        publication_started = finalization_started = False

        def release(text: str, stage: str):
            result = self.guard.release(text)
            transcript.append(Message("user", result.text))
            # Audit metadata never becomes cloud prompt content. Original
            # outgoing drafts remain in this private event log only.
            self.store.save_event("release", {"round": round_id, "stage": stage,
                                             "blocked": result.blocked, "private_draft": text})

        def append_cloud(action):
            transcript.append(Message("assistant", json.dumps(asdict(action), ensure_ascii=False)))

        def finish(status: str, reason: str, after: Harness = base,
                   changed: tuple[str, ...] = ()) -> RoundResult:
            nonlocal finalization_started
            finalization_started = True
            result = RoundResult(round_id, status, reason, base.id, after.id,
                                 inquiries, clarifications, changed)
            self.store.save_record("rounds", round_id, {
                "input_id": input_id, "result": asdict(result),
                "transcript": [asdict(m) for m in transcript]})
            self.store.save_event("round_finished", asdict(result))
            return result

        try:
            release(self.local.prepare(context), "prepare")
            while True:
                remaining = self.config.inquiry_budget - inquiries
                action = self.cloud.decide(tuple(transcript), remaining)
                action.validate(("ASK", "PROPOSE", "RETAIN"))
                append_cloud(action)
                if action.action == "RETAIN":
                    return finish("retained", "cloud_retained")
                if action.action == "PROPOSE":
                    break
                if remaining == 0:
                    raise ProtocolError("Cloud requested inquiry after budget exhaustion")
                inquiries += 1
                release(self.local.investigate(action.content, context), "investigate")

            candidate = Candidate(base)
            # BR clarification exchanges permit BR+1 attempts at realization.
            while True:
                remaining = self.config.clarification_budget - clarifications
                result = self.local.realize(candidate, tuple(transcript), context, remaining)
                if not isinstance(result, Realization) or result.status not in ("READY", "CLARIFY", "STOP"):
                    raise ProtocolError("Invalid realization status")
                if result.status == "READY":
                    if result.reviewed is not True or not result.review.strip():
                        return finish("retained", "artifact_review_missing")
                    checked_id = candidate.freeze().id
                    checks = self.checks.verify(candidate)
                    self.store.save_event("candidate_checks", {
                        "round": round_id, "candidate": checked_id, "review": result.review,
                        "checks": [asdict(c) for c in checks]})
                    if any(not c.passed for c in checks) or candidate.freeze().id != checked_id:
                        return finish("retained", "required_checks_failed")
                    changed = tuple(candidate.changed_paths)
                    if not changed:
                        return finish("retained", "no_file_changes")
                    publication_started = True
                    deployed = self.store.publish(candidate, expected_id=base.id)
                    return finish("published", "ready_and_checked", deployed, changed)
                if result.status == "STOP":
                    return finish("retained", "local_stopped")
                if remaining == 0:
                    return finish("retained", "clarification_budget_exhausted")
                if not result.clarification.strip():
                    raise ProtocolError("Empty clarification request")
                release(result.clarification, "clarification")

                # The prose allows follow-up investigations during realization.
                # They consume the SAME BI pool; each cloud response consumes
                # a BR exchange. No unbounded third loop is introduced.
                while True:
                    if clarifications >= self.config.clarification_budget:
                        return finish("retained", "followup_unfinished_at_limit")
                    clarifications += 1
                    remaining_inquiries = self.config.inquiry_budget - inquiries
                    reply = self.cloud.respond(tuple(transcript), remaining_inquiries)
                    reply.validate(("RESPOND", "PROPOSE", "ASK", "WITHDRAW", "RETAIN"))
                    append_cloud(reply)
                    if reply.action in ("WITHDRAW", "RETAIN"):
                        return finish("retained", "proposal_withdrawn")
                    if reply.action != "ASK":
                        break
                    if remaining_inquiries == 0:
                        raise ProtocolError("Follow-up inquiry budget exhausted")
                    inquiries += 1
                    release(self.local.investigate(reply.content, context, candidate), "followup")
        except Exception as exc:
            if publication_started or finalization_started:
                # A filesystem error after CAS can mean publication succeeded.
                # Do not misreport the previous harness or repeat model calls.
                raise PendingRound("Publication/completion requires inspection; no retry was attempted") from exc
            # Fail closed; never turn an incomplete call into a local fallback
            # update. Private diagnostics remain in the local store.
            self.store.save_event("round_error", {"round": round_id,
                                                  "type": type(exc).__name__, "detail": str(exc)})
            return finish("failed", type(exc).__name__)
