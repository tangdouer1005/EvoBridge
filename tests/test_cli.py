from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evobridge import Harness, RunConfig, SnapshotStore
from evobridge.cli import main



class CLIIntegrationTests(unittest.TestCase):
    def test_offline_cli_uses_real_role_protocol_and_saves_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            harness = root / "initial"
            Harness({"memory.md": "", "CONTEXT.md": "Read memory.md."}).write_to(harness)
            config = root / "config.json"
            config.write_text(json.dumps(asdict(RunConfig())))
            policy = root / "policy.json"
            policy.write_text(json.dumps({"prohibited_patterns": ["PRIVATE_SENTINEL"]}))
            corpus = root / "corpus.jsonl"
            corpus.write_text(json.dumps({"id": "one", "task": "Private task", "outcome": False,
                                          "observations": ["PRIVATE_SENTINEL"], "feedback": "Original"}) + "\n")
            requests = []

            @dataclass
            class FakeUsage:
                input_tokens: int = 100
                output_tokens: int = 10
                attempted: bool = True

            @dataclass
            class FakeSession:
                status: str = "completed"

            class FakeRuntime:
                def __init__(self, base_url, settings, **kwargs):
                    self.settings = settings
                    self.role = "local" if "local" in base_url else "cloud"
                    self.runtime_identity = {"backend": "test-native", "endpoint": base_url}
                    self.usage_receipts = []
                    self.session_receipts = []

                def run(self, instructions, prompt, tools, dispatch):
                    requests.append((self.role, {"instructions": instructions, "prompt": prompt,
                                                 "tools": tools}))
                    self.usage_receipts.append(FakeUsage())
                    self.session_receipts.append(FakeSession())
                    if self.role == "cloud":
                        assert tools == []
                        return json.dumps({"action": "PROPOSE", "content": "Update locally verified domain memory."})
                    if "PREPARATION" in instructions:
                        return json.dumps({"content": "PRIVATE_SENTINEL in proposed brief"})
                    dispatch("write", {"path": "memory.md", "content": "Scoped fact"})
                    return json.dumps({"status": "READY", "reviewed": True,
                                       "review": "Checked applicability, consistency and links."})

            args = ["offline", "--workdir", str(root / "run"), "--config", str(config),
                    "--policy", str(policy), "--harness", str(harness), "--corpus", str(corpus),
                    "--epochs", "1", "--local-base-url", "http://local/v1",
                    "--cloud-base-url", "http://cloud/v1"]
            output = io.StringIO()
            with patch("evobridge.runtime.CodexRuntime", FakeRuntime), redirect_stdout(output):
                self.assertEqual(main(args), 0)
            store = SnapshotStore(root / "run")
            self.assertEqual(store.current().files["memory.md"], "Scoped fact")
            self.assertEqual(store.history("rounds")[0]["result"]["status"], "published")
            cloud_payload = json.dumps([body for role, body in requests if role == "cloud"])
            self.assertNotIn("PRIVATE_SENTINEL", cloud_payload)
            self.assertNotIn("Private task", cloud_payload)
            self.assertIn("BLOCKED", cloud_payload)
            events = [json.loads(line) for line in (root / "run" / "events.jsonl").read_text().splitlines()]
            usage = [event["data"] for event in events if event["kind"] == "provider_usage"]
            self.assertEqual(len(usage), 3)
            self.assertEqual(sum(u["input_tokens"] for u in usage), 300)
            self.assertTrue(all(u["attempted"] for u in usage))
            self.assertEqual({u["role"] for u in usage}, {"local", "cloud"})
            # Reopening a completed corpus does not call either model again.
            with patch("evobridge.runtime.CodexRuntime", FakeRuntime), redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(len(requests), 3)

    def test_init_does_not_overwrite_an_existing_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory).resolve() / "harness")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["init", "--directory", target]), 0)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["init", "--directory", target]), 1)
            self.assertTrue((Path(target) / "CONTEXT.md").exists())


if __name__ == "__main__":
    unittest.main()
