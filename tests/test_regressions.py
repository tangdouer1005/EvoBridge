from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evobridge import (Candidate, CheckResult, CheckSuite, Engine, Experience,
                       Harness, ModelSettings, PolicyGuard, RegisteredCheck,
                       RunConfig, Runner, SnapshotStore, Task, evaluate_frozen)
from evobridge.demo import DemoCloud, DemoLocal, DemoWorker


class RuntimeRegressionTests(unittest.TestCase):
    def test_mutating_check_cannot_smuggle_edits_into_candidate(self):
        base = Harness({"memory.md": "original", "CONTEXT.md": "Read memory.md."})
        candidate = Candidate(base)
        candidate.write("CONTEXT.md", "Updated policy")
        observed = []
        def impure(view):
            observed.append((view.base_id, view.changed_paths))
            view.write("memory.md", "check-created edit")
            return CheckResult("check", True)
        suite = CheckSuite({"check": RegisteredCheck(impure)})
        self.assertFalse(suite.run_one("check", candidate).passed)
        self.assertFalse(suite.verify(candidate)[0].passed)
        self.assertEqual(candidate.read("memory.md"), "original")
        self.assertEqual(observed, [(base.id, ("CONTEXT.md",))] * 2)

    def test_check_exception_cannot_leave_mutated_candidate(self):
        candidate = Candidate(Harness({"memory.md": "original", "CONTEXT.md": "policy"}))
        def fails(view):
            view.write("memory.md", "injected")
            raise RuntimeError("failed")
        result = CheckSuite({"fail": RegisteredCheck(fails)}).run_one("fail", candidate)
        self.assertFalse(result.passed)
        self.assertEqual(candidate.read("memory.md"), "original")

    def test_frozen_evaluation_refuses_changed_sampling_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(Path(directory).resolve())
            harness = Harness({"memory.md": "", "CONTEXT.md": "policy"})
            worker = DemoWorker()
            worker.client = SimpleNamespace(settings=ModelSettings.local())
            tasks = [Task("same", "Task")]
            evaluate_frozen(worker, tasks, harness, store=store)
            worker.client = SimpleNamespace(settings=replace(ModelSettings.local(), temperature=1.5))
            with self.assertRaises(ValueError):
                evaluate_frozen(worker, tasks, harness, store=store)

    def test_role_sampling_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            local = DemoLocal()
            local.client = SimpleNamespace(settings=replace(ModelSettings.local(), temperature=1.5))
            store = SnapshotStore(Path(directory).resolve())
            with self.assertRaisesRegex(ValueError, "inference settings"):
                Engine(store, local, DemoCloud(), PolicyGuard([]))

    def test_worker_sampling_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(SnapshotStore(Path(directory).resolve()), DemoLocal(), DemoCloud(), PolicyGuard([]))
            worker = DemoWorker()
            worker.client = SimpleNamespace(settings=replace(ModelSettings.local(), temperature=1.5))
            with self.assertRaises(ValueError):
                Runner(engine, worker)


if __name__ == "__main__":
    unittest.main()
