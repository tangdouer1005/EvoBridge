"""Independent checks of the final paper's externally visible contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evobridge.checks import CheckSuite, RegisteredCheck
from evobridge.config import ModelSettings, RunConfig
from evobridge.engine import Engine, PendingRound
from evobridge.harness import Harness
from evobridge.policy import PolicyGuard
from evobridge.schedules import Runner, evaluate_frozen, offline
from evobridge.storage import SnapshotStore
from evobridge.types import CheckResult, CloudAction, Experience, Realization, Task


class PolicyBoundaryContractTests(unittest.TestCase):
    def test_prohibited_match_replaces_the_entire_message(self):
        guard = PolicyGuard([r"ACCOUNT-\d+", r"(?i)confidential"])
        for message in (
            "Harmless prefix ACCOUNT-12345 harmless suffix",
            "Facts:\nCONFIDENTIAL business material\nSuggested fix",
        ):
            with self.subTest(message=message):
                released = guard.release(message)
                self.assertEqual(released.text, "BLOCKED")
                self.assertTrue(released.blocked)

    def test_allowed_messages_are_forwarded_without_rewriting(self):
        guard = PolicyGuard([r"ACCOUNT-\d+"])
        messages = ("", "  leading and trailing whitespace  ", "规则\nA\tB\r\n")
        for message in messages:
            with self.subTest(message=message):
                released = guard.release(message)
                self.assertEqual(released.text, message)
                self.assertFalse(released.blocked)

    def test_only_explicit_text_crosses_the_boundary(self):
        guard = PolicyGuard([])
        for private_object in ({"history": "secret"}, ["secret"], b"secret", None):
            with self.subTest(kind=type(private_object).__name__):
                with self.assertRaises(TypeError):
                    guard.release(private_object)


class AppendixConfigurationContractTests(unittest.TestCase):
    def test_local_profiles_match_table_five(self):
        shared = {
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1.0,
            "thinking": False,
            "context_tokens": 131072,
            "compact_at": 98304,
            "max_output_tokens": 32768,
            "tool_output_tokens": 8192,
        }
        for size, model, temperature in (
            ("9b", "Qwen3.5-9B", 0.6),
            ("27b", "Qwen3.8-27B", 1.0),
        ):
            with self.subTest(size=size):
                profile = asdict(ModelSettings.local(size))
                self.assertEqual(profile["model"], model)
                self.assertEqual(profile["temperature"], temperature)
                self.assertEqual({key: profile[key] for key in shared}, shared)

    def test_default_cloud_and_exchange_limits_match_appendix(self):
        config = RunConfig()
        self.assertEqual(config.cloud.model, "gpt6-sol")
        self.assertEqual(config.cloud.reasoning_effort, "medium")
        self.assertEqual((config.inquiry_budget, config.clarification_budget), (3, 3))

    def test_runtime_configuration_is_fixed(self):
        config = RunConfig()
        with self.assertRaises(FrozenInstanceError):
            config.local = ModelSettings.local("27b")
        with self.assertRaises(FrozenInstanceError):
            config.local.model = "another-model"

    def test_zero_budgets_are_legal_but_negative_or_nonintegers_are_not(self):
        RunConfig(inquiry_budget=0, clarification_budget=0)
        for field in ("inquiry_budget", "clarification_budget"):
            for value in (-1, True, 1.5, "3"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        RunConfig(**{field: value})


class _Local:
    model_id = "Qwen3.5-9B"

    def __init__(self, realizations=(), *, private_text="Permitted factual brief"):
        self.realizations = iter(realizations)
        self.private_text = private_text
        self.investigations = []
        self.attempts = []
        self.prepare_hook = None
        self.realize_hook = None

    def prepare(self, context):
        if self.prepare_hook:
            self.prepare_hook(context)
        return self.private_text

    def investigate(self, question, context, candidate=None):
        self.investigations.append((question, candidate is not None))
        return self.private_text

    def realize(self, candidate, transcript, context, remaining):
        self.attempts.append((id(candidate), remaining))
        if self.realize_hook:
            self.realize_hook(candidate, transcript, context)
        candidate.write("memory.md", f"reviewed revision {len(self.attempts)}")
        return next(self.realizations)


class _Cloud:
    model_id = "gpt6-sol"

    def __init__(self, decisions=(), replies=()):
        self.decisions, self.replies = iter(decisions), iter(replies)
        self.decision_calls, self.response_calls = [], []

    def decide(self, transcript, remaining):
        self.decision_calls.append((transcript, remaining))
        return next(self.decisions)

    def respond(self, transcript, remaining_inquiries):
        self.response_calls.append((transcript, remaining_inquiries))
        return next(self.replies)


def _ready():
    return Realization("READY", reviewed=True,
                       review="Applicability, consistency and capability links reviewed.")


class AlgorithmOneContractTests(unittest.TestCase):
    def make_engine(self, local, cloud, *, bi=3, br=3, checks=None, guard=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = SnapshotStore(Path(temporary.name).resolve())
        initial = Harness({"memory.md": "old memory", "CONTEXT.md": "load relevant memory"})
        engine = Engine(store, local, cloud, guard or PolicyGuard([]), checks,
                        RunConfig(inquiry_budget=bi, clarification_budget=br), initial=initial)
        return engine, initial

    @staticmethod
    def experience():
        return Experience("completed-task", "PRIVATE-CANARY task", {"success": False},
                          observations=("PRIVATE-CANARY trace",),
                          feedback="PRIVATE-CANARY feedback")

    def test_inquiry_limit_counts_exchanges_not_decisions(self):
        for budget in (0, 1, 3):
            with self.subTest(budget=budget):
                local = _Local([_ready()])
                cloud = _Cloud([CloudAction("ASK", "Find evidence")] * budget +
                               [CloudAction("PROPOSE", "Revise memory applicability")])
                engine, initial = self.make_engine(local, cloud, bi=budget, br=0)
                result = engine.evolve(self.experience())
                self.assertEqual(result.status, "published")
                self.assertEqual(result.inquiries, budget)
                self.assertEqual(len(local.investigations), budget)
                self.assertEqual([n for _, n in cloud.decision_calls],
                                 list(range(budget, -1, -1)))
                self.assertNotEqual(engine.store.current().id, initial.id)

    def test_ready_on_final_realization_attempt_still_publishes(self):
        for budget in (0, 1, 3):
            with self.subTest(budget=budget):
                local = _Local([Realization("CLARIFY", "Which applicability scope?")] * budget +
                               [_ready()])
                cloud = _Cloud([CloudAction("PROPOSE", "Revise memory applicability")],
                               [CloudAction("RESPOND", "Use the narrow scope")] * budget)
                engine, initial = self.make_engine(local, cloud, bi=0, br=budget)

                def check_isolation(candidate, transcript, context):
                    self.assertEqual(engine.store.current().id, initial.id)
                    self.assertEqual(context.harness.id, initial.id)

                local.realize_hook = check_isolation
                result = engine.evolve(self.experience())
                self.assertEqual(result.status, "published")
                self.assertEqual(result.clarifications, budget)
                self.assertEqual(len(cloud.response_calls), budget)
                self.assertEqual([n for _, n in local.attempts], list(range(budget, -1, -1)))
                self.assertEqual(len({candidate for candidate, _ in local.attempts}), 1)
                self.assertEqual(engine.store.current().files["memory.md"],
                                 f"reviewed revision {budget + 1}")

    def test_unfinished_candidate_at_limit_never_publishes(self):
        for budget in (0, 1, 3):
            with self.subTest(budget=budget):
                local = _Local([Realization("CLARIFY", "Still unresolved")] * (budget + 1))
                cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")],
                               [CloudAction("RESPOND", "Clarification")] * budget)
                engine, initial = self.make_engine(local, cloud, br=budget)
                result = engine.evolve(self.experience())
                self.assertEqual(result.status, "retained")
                self.assertEqual(len(local.attempts), budget + 1)
                self.assertEqual(len(cloud.response_calls), budget)
                self.assertEqual(engine.store.current().id, initial.id)

    def test_all_local_cloud_paths_are_guarded_including_followup(self):
        local = _Local([Realization("CLARIFY", "PRIVATE-CANARY condition"), _ready()],
                       private_text="PRIVATE-CANARY evidence")
        cloud = _Cloud([CloudAction("ASK", "Inspect local evidence"),
                        CloudAction("PROPOSE", "Revise memory")],
                       [CloudAction("ASK", "Inspect candidate behavior"),
                        CloudAction("RESPOND", "Apply the scoped change")])
        engine, _ = self.make_engine(local, cloud, bi=2, br=2,
                                     guard=PolicyGuard(["PRIVATE-CANARY"]))
        result = engine.evolve(self.experience())
        self.assertEqual(result.status, "published")
        self.assertEqual((result.inquiries, result.clarifications), (2, 2))
        self.assertEqual([is_followup for _, is_followup in local.investigations], [False, True])
        for transcript, _ in cloud.decision_calls + cloud.response_calls:
            for message in transcript:
                self.assertNotIn("PRIVATE-CANARY", message.content)
                if message.role == "user":
                    self.assertEqual(message.content, "BLOCKED")
        final_transcript = engine.store.read_record("rounds", "completed-task")["transcript"]
        self.assertEqual([m["content"] for m in final_transcript if m["role"] == "user"],
                         ["BLOCKED"] * 4)

    def test_exhausted_or_malformed_cloud_decisions_cannot_publish(self):
        for action in (CloudAction("ASK", "One more inquiry"),
                       CloudAction("UNKNOWN", "Invalid action"),
                       CloudAction("PROPOSE", "")):
            with self.subTest(action=action):
                local = _Local([_ready()])
                engine, initial = self.make_engine(local, _Cloud([action]), bi=0)
                result = engine.evolve(self.experience())
                self.assertEqual(result.status, "failed")
                self.assertFalse(local.investigations)
                self.assertFalse(local.attempts)
                self.assertEqual(engine.store.current().id, initial.id)

    def test_followup_cannot_reset_the_global_inquiry_budget(self):
        local = _Local([Realization("CLARIFY", "Need design detail")])
        cloud = _Cloud([CloudAction("ASK", "Use the only investigation"),
                        CloudAction("PROPOSE", "Revise memory")],
                       [CloudAction("ASK", "Ask beyond global allowance")])
        engine, initial = self.make_engine(local, cloud, bi=1, br=3)
        result = engine.evolve(self.experience())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.inquiries, 1)
        self.assertEqual(len(local.investigations), 1)
        self.assertEqual(cloud.response_calls[0][1], 0)
        self.assertEqual(engine.store.current().id, initial.id)

    def test_followup_responses_share_the_clarification_cap(self):
        local = _Local([Realization("CLARIFY", "Need more evidence")])
        cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")],
                       [CloudAction("ASK", "Follow-up question")])
        engine, initial = self.make_engine(local, cloud, bi=3, br=1)
        result = engine.evolve(self.experience())
        self.assertEqual(result.status, "retained")
        self.assertEqual((result.inquiries, result.clarifications), (1, 1))
        self.assertEqual(len(cloud.response_calls), 1)
        self.assertEqual(engine.store.current().id, initial.id)

    def test_checks_and_review_gate_adoption_not_the_primary_score(self):
        for reviewed, passed, expected in ((True, True, "published"),
                                          (True, False, "retained"),
                                          (False, True, "retained")):
            with self.subTest(reviewed=reviewed, passed=passed):
                local = _Local([Realization("READY", reviewed=reviewed, review="Artifact reviewed")])
                checks = CheckSuite({"local_contract": RegisteredCheck(
                    lambda candidate, passed=passed: CheckResult("local_contract", passed))})
                engine, initial = self.make_engine(
                    local, _Cloud([CloudAction("PROPOSE", "Revise memory")]), checks=checks)
                result = engine.evolve(self.experience())  # The primary task failed.
                self.assertEqual(result.status, expected)
                self.assertEqual(engine.store.current().id == initial.id, expected != "published")
                self.assertEqual(engine.store.read_record("experience", "completed-task")
                                 ["outcome"], {"success": False})

    def test_withdrawal_discards_already_edited_candidate(self):
        local = _Local([Realization("CLARIFY", "Implementation conflicts with proposal")])
        cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")],
                       [CloudAction("WITHDRAW", "Keep existing behavior")])
        engine, initial = self.make_engine(local, cloud)
        result = engine.evolve(self.experience())
        self.assertEqual(result.status, "retained")
        self.assertEqual(len(local.attempts), 1)
        self.assertEqual(engine.store.current().id, initial.id)

    def test_reusing_a_completed_round_with_changed_explicit_history_is_rejected(self):
        local = _Local([_ready()])
        cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")])
        engine, _ = self.make_engine(local, cloud)
        experience = self.experience()
        first = (Experience("earlier-task", "Earlier task", {"success": False}),)
        changed = (Experience("earlier-task", "Earlier task", {"success": True}),)
        result = engine.evolve(experience, round_id="same-round", history=first)
        self.assertEqual(engine.evolve(experience, round_id="same-round", history=first), result)
        with self.assertRaises(ValueError):
            engine.evolve(experience, round_id="same-round", history=changed)
        self.assertEqual(len(cloud.decision_calls), 1)
        self.assertEqual(len(local.attempts), 1)

    def test_error_after_publication_never_records_failed_with_the_old_harness(self):
        local = _Local([_ready()])
        cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")])
        engine, initial = self.make_engine(local, cloud)
        actual_publish = engine.store.publish

        def commit_then_interrupt(*args, **kwargs):
            actual_publish(*args, **kwargs)
            raise OSError("Simulated interrupted acknowledgement after committed publication")

        with patch.object(engine.store, "publish", side_effect=commit_then_interrupt):
            with self.assertRaises(PendingRound):
                engine.evolve(self.experience())
        self.assertNotEqual(engine.store.current().id, initial.id)
        self.assertIsNone(engine.store.read_record("rounds", "completed-task"))
        with self.assertRaises(PendingRound):
            engine.evolve(self.experience())
        self.assertEqual(len(cloud.decision_calls), 1)
        self.assertEqual(len(local.attempts), 1)

    def test_failed_final_record_write_leaves_published_round_pending(self):
        local = _Local([_ready()])
        cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")])
        engine, initial = self.make_engine(local, cloud)
        actual_save = engine.store.save_record

        def interrupt_round_record(kind, key, data):
            if kind == "rounds":
                raise OSError("Simulated final record write failure")
            return actual_save(kind, key, data)

        with patch.object(engine.store, "save_record", side_effect=interrupt_round_record):
            with self.assertRaises(PendingRound):
                engine.evolve(self.experience())
        self.assertNotEqual(engine.store.current().id, initial.id)
        self.assertIsNone(engine.store.read_record("rounds", "completed-task"))
        with self.assertRaises(PendingRound):
            engine.evolve(self.experience())
        self.assertEqual(len(cloud.decision_calls), 1)

    def test_error_after_final_record_keeps_the_accurate_published_result(self):
        local = _Local([_ready()])
        cloud = _Cloud([CloudAction("PROPOSE", "Revise memory")])
        engine, initial = self.make_engine(local, cloud)
        actual_event = engine.store.save_event

        def interrupt_final_event(kind, data):
            if kind == "round_finished":
                raise OSError("Simulated audit event write failure")
            return actual_event(kind, data)

        with patch.object(engine.store, "save_event", side_effect=interrupt_final_event):
            with self.assertRaises(PendingRound):
                engine.evolve(self.experience())
        saved = engine.store.read_record("rounds", "completed-task")["result"]
        self.assertEqual(saved["status"], "published")
        self.assertEqual(saved["before"], initial.id)
        self.assertEqual(saved["after"], engine.store.current().id)
        replay = engine.evolve(self.experience())
        self.assertEqual(replay.status, "published")
        self.assertEqual(replay.after, engine.store.current().id)
        self.assertEqual(len(cloud.decision_calls), 1)

    def test_executable_edits_require_a_passing_registered_behavioral_check(self):
        for behavioral, passed, expected in ((False, True, "retained"),
                                             (True, False, "retained"),
                                             (True, True, "published")):
            with self.subTest(behavioral=behavioral, passed=passed):
                local = _Local([_ready()])
                local.realize_hook = lambda candidate, transcript, context: candidate.write(
                    "skills/helper.py", "def combine(a, b):\n    return a + b\n")
                checks = CheckSuite({"behavior": RegisteredCheck(
                    lambda candidate, passed=passed: CheckResult("behavior", passed),
                    behavioral=behavioral)})
                engine, initial = self.make_engine(
                    local, _Cloud([CloudAction("PROPOSE", "Add a reusable operation")]),
                    checks=checks)
                result = engine.evolve(self.experience())
                self.assertEqual(result.status, expected)
                self.assertEqual(engine.store.current().id == initial.id, expected != "published")

    def test_truthy_values_are_not_completed_artifact_reviews(self):
        for reviewed, review in ((1, "review"), ("true", "review"), (True, "   ")):
            with self.subTest(reviewed=reviewed, review=review):
                local = _Local([Realization("READY", reviewed=reviewed, review=review)])
                engine, initial = self.make_engine(
                    local, _Cloud([CloudAction("PROPOSE", "Revise memory")]))
                result = engine.evolve(self.experience())
                self.assertEqual(result.status, "retained")
                self.assertEqual(engine.store.current().id, initial.id)


class ScheduleOutcomeContractTests(unittest.TestCase):
    make_engine = AlgorithmOneContractTests.make_engine

    def test_primary_outcome_is_sealed_before_adaptation(self):
        local = _Local([_ready()])
        cloud = _Cloud([CloudAction("PROPOSE", "Learn from the failed primary")])
        engine, initial = self.make_engine(local, cloud)
        worker_calls = []

        class Worker:
            model_id = "Qwen3.5-9B"

            def execute(self, task, harness):
                worker_calls.append(harness.id)
                return Experience(task.id, task.instruction, {"success": False})

        def inspect_primary(context):
            sealed = engine.store.read_record("primary", context.experience.id)
            self.assertIsNotNone(sealed)
            self.assertEqual(sealed["experience"]["outcome"], {"success": False})
            context.experience.outcome["success"] = True  # Cannot rewrite durable primary.

        local.prepare_hook = inspect_primary
        runner = Runner(engine, Worker())
        tasks = [Task("completed-task", "Execute once")]
        runner.online(tasks)
        runner.online(tasks)  # Reload sealed results, no worker rerun.
        self.assertEqual(worker_calls, [initial.id])
        self.assertEqual(engine.store.read_record("primary", "completed-task")
                         ["experience"]["outcome"], {"success": False})

    def test_frozen_execution_cannot_call_the_evolution_roles(self):
        local, cloud = _Local(), _Cloud()
        engine, initial = self.make_engine(local, cloud)
        worker_harnesses = []

        class Worker:
            model_id = "Qwen3.5-9B"

            def execute(self, task, harness):
                worker_harnesses.append(harness.id)
                return Experience(task.id, task.instruction, {"success": True},
                                  feedback="Evaluation-only feedback")

        tasks = [Task("evaluation-one", "Evaluate once")]
        for _ in range(2):
            evaluate_frozen(Worker(), tasks, initial, store=engine.store)
        self.assertEqual(worker_harnesses, [initial.id])
        self.assertFalse(local.attempts)
        self.assertFalse(cloud.decision_calls)
        self.assertFalse(cloud.response_calls)
        self.assertEqual(engine.store.history("experience"), [])

    def test_reset_restores_initial_harness_and_excludes_earlier_block_history(self):
        local = _Local([_ready()] * 3)
        cloud = _Cloud([CloudAction("PROPOSE", "Learn from this task")] * 3)
        engine, initial = self.make_engine(local, cloud)
        worker_harnesses, visible_history = [], []

        class Worker:
            model_id = "Qwen3.5-9B"

            def execute(self, task, harness):
                worker_harnesses.append(harness.id)
                return Experience(task.id, task.instruction, {"success": False})

        local.prepare_hook = lambda context: visible_history.append(
            [experience.id for experience in context.history])
        runner = Runner(engine, Worker())
        tasks = [Task(f"task-{i}", "Execute with the current harness") for i in range(3)]
        runner.online(tasks, reset_every=2)
        self.assertEqual(worker_harnesses[0], initial.id)
        self.assertNotEqual(worker_harnesses[1], initial.id)
        self.assertEqual(worker_harnesses[2], initial.id)
        self.assertEqual(visible_history, [["task-0"], ["task-0", "task-1"], ["task-2"]])
        final_id = engine.store.current().id
        runner.online(tasks, reset_every=2)
        self.assertEqual(engine.store.current().id, final_id)
        self.assertEqual(len(worker_harnesses), 3)

    def test_offline_revisits_fixed_records_and_preserves_epoch_zero(self):
        local = _Local([_ready()] * 4)
        cloud = _Cloud([CloudAction("PROPOSE", "Refine reusable applicability")] * 4)
        engine, initial = self.make_engine(local, cloud)
        visited = []
        local.prepare_hook = lambda context: visited.append(context.experience.id)
        corpus = [Experience("history-a", "Old task A", {"success": False}),
                  Experience("history-b", "Old task B", {"success": True})]
        checkpoints = offline(engine, corpus, epochs=2, checkpoints=(0, 1, 2))
        self.assertEqual(visited, ["history-a", "history-b"] * 2)
        self.assertEqual(checkpoints[0].id, initial.id)
        self.assertEqual(checkpoints[1].files["memory.md"], "reviewed revision 2")
        self.assertEqual(checkpoints[2].files["memory.md"], "reviewed revision 4")
        self.assertEqual(engine.store.history("primary"), [])
        self.assertEqual(len(engine.store.history("experience")), 2)
        offline(engine, corpus, epochs=2, checkpoints=(0, 1, 2))
        self.assertEqual(len(visited), 4)


if __name__ == "__main__":
    unittest.main()
