import json
from pathlib import Path
import tempfile
import unittest

from evobridge import (Candidate, CloudAction, Engine, Experience, Harness,
                       ModelSettings, PendingRound, PolicyGuard, Realization,
                       RunConfig, Runner, SnapshotStore, Task, evaluate_frozen,
                       offline, transfer)


class Local:
    model_id = "Qwen3.5-9B"

    def __init__(self):
        self.contexts = []

    def prepare(self, ctx):
        self.contexts.append(ctx)
        return "Completed authorized experience."

    def investigate(self, q, ctx, candidate=None):
        return "Evidence."

    def realize(self, candidate, transcript, ctx, remaining):
        candidate.write("memory.md", candidate.read("memory.md") + ctx.experience.id + "\n")
        return Realization("READY", reviewed=True, review="Applicability, consistency and links reviewed.")


class Cloud:
    model_id = "gpt6-sol"

    def decide(self, transcript, remaining):
        return CloudAction("PROPOSE", "Consolidate relevant domain memory.")

    def respond(self, transcript, remaining):
        raise AssertionError("No clarification expected")


class Worker:
    model_id = "Qwen3.5-9B"

    def __init__(self):
        self.calls = []

    def execute(self, task, harness):
        self.calls.append((task.id, dict(harness.files)))
        return Experience(task.id, task.instruction, {"success": False})


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.initial = Harness({"memory.md": "", "CONTEXT.md": "Read memory.md."})
        self.store = SnapshotStore(self.root)
        self.local = Local()
        self.engine = Engine(self.store, self.local, Cloud(), PolicyGuard([]), initial=self.initial)

    def test_online_seals_before_adaptation_and_reuses_records(self):
        worker = Worker()
        runner = Runner(self.engine, worker)
        tasks = [Task("a", "First"), Task("b", "Second")]
        runner.online(tasks)
        self.assertEqual(worker.calls[0][1]["memory.md"], "")
        self.assertEqual(worker.calls[1][1]["memory.md"], "a\n")
        events = [json.loads(s)["kind"] for s in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertLess(events.index("primary_recorded"), events.index("release"))
        self.assertEqual(self.store.read_record("primary", "a")["experience"]["outcome"], {"success": False})
        runner.online(tasks)
        self.assertEqual(len(worker.calls), 2)
        with self.assertRaises(ValueError):
            runner.online(list(reversed(tasks)))

    def test_local_payload_mutation_does_not_change_sealed_primary(self):
        original = self.local.prepare
        def prepare(ctx):
            ctx.experience.outcome["success"] = True
            return original(ctx)
        self.local.prepare = prepare
        result = Runner(self.engine, Worker()).online([Task("a", "First")])
        self.assertFalse(result[0].outcome["success"])
        self.assertFalse(self.store.read_record("primary", "a")["experience"]["outcome"]["success"])

    def test_reset_discards_previous_block_harness_and_accessible_history(self):
        worker = Worker()
        tasks = [Task(str(i), "Work") for i in range(5)]
        Runner(self.engine, worker).online(tasks, reset_every=2)
        self.assertEqual(worker.calls[2][1]["memory.md"], "")
        self.assertEqual(worker.calls[4][1]["memory.md"], "")
        self.assertEqual([e.id for e in self.local.contexts[2].history], ["2"])
        self.assertEqual([e.id for e in self.local.contexts[3].history], ["2", "3"])

    def test_offline_revisits_same_corpus_and_preserves_checkpoint_zero(self):
        corpus = [Experience("a", "Fixed", True), Experience("b", "Fixed", False)]
        snapshots = offline(self.engine, corpus, epochs=3)
        self.assertEqual(snapshots[0].id, self.initial.id)
        self.assertEqual(snapshots[3].files["memory.md"], "a\nb\n" * 3)
        self.assertEqual(len(self.local.contexts), 6)
        offline(self.engine, corpus, epochs=3)
        self.assertEqual(len(self.local.contexts), 6)
        self.assertEqual(len(self.store.history("experience")), 2)
        with self.assertRaises(ValueError):
            offline(self.engine, list(reversed(corpus)), epochs=3)

    def test_modes_cannot_share_adaptation_state(self):
        offline(self.engine, [], epochs=0)
        with self.assertRaises(ValueError):
            Runner(self.engine, Worker()).online([])

    def test_frozen_evaluation_never_enters_adaptation_history(self):
        worker = Worker()
        tasks = [Task("eval", "Do not learn this")]
        evaluate_frozen(worker, tasks, self.initial, store=self.store)
        evaluate_frozen(worker, tasks, self.initial, store=self.store)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(self.local.contexts, [])
        self.assertEqual(self.store.history("experience"), [])
        self.assertEqual(self.store.current().id, self.initial.id)

    def test_transfer_creates_fresh_target_workers(self):
        created = []
        def factory(target):
            worker = Worker()
            created.append((target, worker))
            return worker
        output = transfer({"A": self.initial, "B": self.initial},
                          {"X": [Task("x", "X")], "Y": [Task("y", "Y")]}, factory)
        self.assertEqual(len(created), 4)
        self.assertEqual(set(output), {"A", "B"})
        self.assertEqual(self.store.history("experience"), [])

    def test_interrupted_primary_is_not_reexecuted(self):
        worker = Worker()
        self.store.save_record("primary-starts", "a", {"task": "already called"})
        with self.assertRaises(PendingRound):
            Runner(self.engine, worker).online([Task("a", "Do work")])
        self.assertEqual(worker.calls, [])

    def test_worker_checkpoint_cannot_change(self):
        worker = Worker()
        worker.model_id = "different"
        with self.assertRaises(ValueError):
            Runner(self.engine, worker)


if __name__ == "__main__":
    unittest.main()
