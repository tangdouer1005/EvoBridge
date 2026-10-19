"""Persistence and boundary tests, independent of models and benchmarks."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

from evobridge.harness import Candidate, Harness
from evobridge.storage import SnapshotStore


BASE = {"memory.md": "Domain facts.\n", "CONTEXT.md": "Read relevant skills.\n",
        "skills/example/SKILL.md": "A reusable procedure.\n"}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)

    def test_harness_is_immutable_and_hash_is_order_independent(self):
        original = dict(BASE)
        harness = Harness(original)
        original["memory.md"] = "Mutated input"
        self.assertEqual(harness.files["memory.md"], BASE["memory.md"])
        self.assertEqual(harness.id, Harness(dict(reversed(list(BASE.items())))).id)
        with self.assertRaises(TypeError):
            harness.files["memory.md"] = "Mutation"

    def test_candidate_copy_tracks_add_edit_delete_and_reversion(self):
        harness = Harness(BASE)
        candidate = Candidate(harness)
        candidate.write("memory.md", "New fact")
        candidate.write("skills/example/helper.py", "print('example')\n")
        candidate.delete("skills/example/SKILL.md")
        self.assertEqual(candidate.changed_paths,
                         ("memory.md", "skills/example/SKILL.md", "skills/example/helper.py"))
        self.assertEqual(harness.files, BASE)
        frozen = candidate.freeze()
        candidate.write("memory.md", BASE["memory.md"])
        self.assertEqual(frozen.files["memory.md"], "New fact")
        self.assertNotIn("memory.md", candidate.changed_paths)
        self.assertEqual(candidate.list_files(), tuple(sorted(candidate.freeze().files)))

    def test_paths_required_roots_and_text_types_are_enforced(self):
        candidate = Candidate(Harness(BASE))
        for path in ("../escape", "/tmp/escape", "skills/../escape", "skills//x",
                     "skills/./x", "skills/.secret", "skills/x/", "skills\\x", "config.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                candidate.write(path, "bad")
        for content in (b"binary", "text\0binary", "\ud800"):
            with self.subTest(content=repr(content)), self.assertRaises(ValueError):
                candidate.write("memory.md", content)
        with self.assertRaises(ValueError):
            Harness({"memory.md": "Missing policy"})
        with self.assertRaises(ValueError):
            candidate.delete("CONTEXT.md")
        with self.assertRaises(ValueError):
            Harness({**BASE, "skills/collision": "file", "skills/collision/child": "file"})
        self.assertEqual(candidate.freeze().files, BASE)

    def test_directory_round_trip_preserves_exact_text(self):
        harness = Harness({**BASE, "memory.md": "中文\r\nline two\n"})
        destination = self.root / "harness"
        harness.write_to(destination)
        self.assertEqual(Harness.from_directory(destination).id, harness.id)
        harness.write_to(destination)
        self.assertEqual(Harness.from_directory(destination).files, harness.files)

    def test_directory_rejects_links_and_unexpected_files(self):
        harness = Harness(BASE)
        destination = self.root / "harness"
        harness.write_to(destination)
        outside = self.root / "outside"
        outside.write_text("Do not overwrite", encoding="utf-8")
        (destination / "memory.md").unlink()
        (destination / "memory.md").symlink_to(outside)
        for operation in (lambda: Harness.from_directory(destination), lambda: harness.write_to(destination)):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(outside.read_text(), "Do not overwrite")
        (destination / "memory.md").unlink()
        harness.write_to(destination)
        (destination / "extra.txt").write_text("Unowned")
        with self.assertRaises(ValueError):
            harness.write_to(destination)
        with self.assertRaises(ValueError):
            Harness.from_directory(destination)

    def test_directory_rejects_symlink_directories_and_hardlinks(self):
        harness = Harness(BASE)
        destination = self.root / "harness"
        harness.write_to(destination)
        (destination / "skills/linked").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            Harness.from_directory(destination)
        (destination / "skills/linked").unlink()
        outside = self.root / "outside"
        os.link(destination / "memory.md", outside)
        with self.assertRaises(ValueError):
            Harness.from_directory(destination)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.harness = Harness(BASE)
        self.store = SnapshotStore(self.root / "store")
        self.store.initialize(self.harness, "frozen-config")

    def test_publish_cas_preserves_old_snapshot_and_initial_identity(self):
        candidate = Candidate(self.harness)
        candidate.write("memory.md", "Updated")
        updated = self.store.publish(candidate, self.harness.id)
        self.assertEqual(self.store.current().id, updated.id)
        self.assertTrue((self.store.root / "snapshots" / f"{self.harness.id}.json").exists())
        self.assertEqual(self.store.initialize(self.harness, "frozen-config").id, updated.id)
        with self.assertRaises(ValueError):
            self.store.publish(Candidate(self.harness), self.harness.id)
        with self.assertRaises(ValueError):
            self.store.publish(Candidate(self.harness), updated.id)
        with self.assertRaises(ValueError):
            self.store.initialize(self.harness, "changed-config")
        with self.assertRaises(ValueError):
            self.store.initialize(updated, "frozen-config")
        self.assertEqual(self.store.current().id, updated.id)

    def test_records_are_immutable_normalized_and_in_insertion_order(self):
        self.assertTrue(self.store.save_record("experience", "z", {"id": "z", "facts": ("first",)}))
        self.store.save_record("experience", "a", {"id": "a"})
        self.assertFalse(self.store.save_record("experience", "z", {"id": "z", "facts": ["first"]}))
        self.assertEqual([r["id"] for r in self.store.history("experience")], ["z", "a"])
        self.assertEqual(self.store.read_record("experience", "z")["facts"], ["first"])
        self.assertIsNone(self.store.read_record("experience", "missing"))
        with self.assertRaises(ValueError):
            self.store.save_record("experience", "z", {"id": "different"})
        with self.assertRaises(ValueError):
            self.store.save_record("../escape", "key", {})
        self.store.save_record("experience", "../../safe-hashed-key", {"id": "safe"})
        self.assertEqual(self.store.history("experience")[-1], {"id": "safe"})

    def test_events_append_and_files_are_private(self):
        self.store.save_event("prepared", {"step": 1})
        self.store.save_event("published", {"step": 2})
        events = [json.loads(line) for line in (self.store.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["data"]["step"] for e in events], [1, 2])
        if os.name != "nt":
            for path in self.store.root.rglob("*"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)

    def test_corrupt_snapshot_is_never_loaded(self):
        snapshot = self.store.root / "snapshots" / f"{self.harness.id}.json"
        snapshot.write_text(json.dumps({"files": {**BASE, "memory.md": "Tampered"}}))
        with self.assertRaises(ValueError):
            self.store.current()

    def test_symlink_storage_inputs_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        linked = self.root / "linked"
        linked.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            SnapshotStore(linked)
        (self.store.root / "records" / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.store.save_record("linked", "key", {"private": "value"})
        self.assertEqual(list(outside.iterdir()), [])

    def test_two_processes_publish_one_winner_and_serialize_records(self):
        script = """
import json, sys
from evobridge.harness import Harness, Candidate
from evobridge.storage import SnapshotStore
store = SnapshotStore(sys.argv[1])
base = Harness(json.loads(sys.argv[2]))
print('claimed' if store.save_record('claims', 'one-call', {'input': 'fixed'}) else 'existing')
candidate = Candidate(base)
candidate.write('memory.md', sys.argv[3])
try:
    store.publish(candidate, base.id)
    print('published')
except ValueError:
    print('stale')
for index in range(12):
    store.save_record('parallel', sys.argv[3] + str(index), {'writer': sys.argv[3], 'index': index})
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        processes = [subprocess.Popen([sys.executable, "-c", script, str(self.store.root),
                                      json.dumps(BASE), name], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for name in ("first", "second")]
        results = [process.communicate(timeout=20) for process in processes]
        for process, (stdout, stderr) in zip(processes, results):
            self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(sorted(stdout.splitlines()[0] for stdout, _ in results), ["claimed", "existing"])
        self.assertEqual(sorted(stdout.splitlines()[1] for stdout, _ in results), ["published", "stale"])
        records = self.store.history("parallel")
        self.assertEqual(len(records), 24)
        for writer in ("first", "second"):
            self.assertEqual([r["index"] for r in records if r["writer"] == writer], list(range(12)))


if __name__ == "__main__":
    unittest.main()
