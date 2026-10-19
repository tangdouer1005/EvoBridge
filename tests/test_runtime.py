"""Native protocol/lifecycle contracts using a real local stub executable.

No test in this file contacts a model provider or the user's Codex account.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest.mock import patch

from evobridge.config import ModelSettings
from evobridge.runtime import CodexRuntime, CodexRuntimeError


TOOL = {"name": "read", "description": "Read a scoped resource", "inputSchema": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}


def event(method, *, thread="thread-1", turn="turn-1", **params):
    return {"method": method, "params": {"threadId": thread, "turnId": turn, **params}}


def message(text="answer", *, phase="final_answer", ident="message-1", **scope):
    item = {"id": ident, "type": "agentMessage", "text": text}
    if phase is not None:
        item["phase"] = phase
    return event("item/completed", item=item, **scope)


def done(status="completed", *, thread="thread-1", turn="turn-1", items=None):
    return {"method": "turn/completed", "params": {"threadId": thread,
        "turn": {"id": turn, "status": status, "items": items or []}}}


def call(*, ident=80, call_id="call-1", thread="thread-1", turn="turn-1",
         name="read", namespace=None, arguments=None):
    return {"id": ident, "method": "item/tool/call", "params": {
        "threadId": thread, "turnId": turn, "tool": name, "namespace": namespace,
        "arguments": {"path": "memory.md"} if arguments is None else arguments,
        "callId": call_id}}


@dataclass
class ProviderReceipt:
    model: str = "model"
    attempted: bool = True
    success: bool = True
    terminal_status: str | None = "completed"
    input_tokens: int | None = 5
    output_tokens: int | None = 3


class FakeProxy:
    instances = []
    request_receipts = []

    def __init__(self, base_url, settings, api_key_env=None, *, allowed_tools=None):
        self.upstream_url, self.settings, self.api_key_env = base_url, settings, api_key_env
        self.allowed_tools = allowed_tools
        self.base_url, self.token = "http://127.0.0.1:5555/v1", "fixture-proxy-token"
        self.receipts = list(type(self).request_receipts)
        self.closed = False
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="evobridge-runtime-tests-")
        self.root = Path(self.temp.name)
        FakeProxy.instances, FakeProxy.request_receipts = [], []
        self.proxy_patch = patch("evobridge.runtime.ResponsesProxy", FakeProxy)
        self.proxy_patch.start()
        self.addCleanup(self.proxy_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def stub(self, actions, *, startup_delay=0, thread_overrides=None):
        trace, metadata = self.root / "trace.jsonl", self.root / "child.json"
        script = self.root / "fake-codex"
        script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""\
            import json, os, pathlib, sys, time
            trace = pathlib.Path({str(trace)!r})
            pathlib.Path({str(metadata)!r}).write_text(json.dumps({{
                'pid': os.getpid(), 'env': dict(os.environ), 'cwd': os.getcwd(), 'argv': sys.argv
            }}))
            def read():
                value = json.loads(sys.stdin.readline())
                with trace.open('a') as f:
                    f.write(json.dumps(value) + '\\n')
                return value
            def send(value):
                print(json.dumps(value), flush=True)
            time.sleep({startup_delay!r})
            init = read()
            send({{'id': init['id'], 'result': {{'userAgent': 'fixture'}}}})
            read()
            thread = read()
            thread_result = {{'thread': {{'id': 'thread-1'}},
                  'model': thread['params']['model'], 'modelProvider': 'evobridge',
                  'approvalPolicy': 'never', 'sandbox': {{'type': 'readOnly'}},
                  'cwd': thread['params']['cwd'], 'reasoningEffort': 'medium'}}
            thread_result.update({thread_overrides or {}!r})
            send({{'id': thread['id'], 'result': thread_result}})
            turn = read()
            for action in {actions!r}:
                kind, value = action
                if kind == 'admit':
                    send({{'id': turn['id'], 'result': {{'turn': {{'id': 'turn-1', 'status': 'inProgress'}}}}}})
                elif kind == 'emit':
                    send(value)
                elif kind == 'receive':
                    read()
                elif kind == 'delay':
                    time.sleep(value)
                elif kind == 'raw':
                    print(value, flush=True)
                elif kind == 'exit':
                    sys.exit(value)
            sys.stdin.read()
            """), encoding="utf-8")
        script.chmod(0o700)
        return script

    def runtime(self, actions=None, **kwargs):
        actions = actions or [("admit", None), ("emit", message()), ("emit", done())]
        return CodexRuntime("http://provider.invalid/v1", ModelSettings("model", reasoning_effort="medium"),
                            "PRIVATE_MODEL_KEY", executable=self.stub(actions), **kwargs)

    def trace(self):
        return [json.loads(line) for line in (self.root / "trace.jsonl").read_text().splitlines()]

    def assert_cleaned(self):
        info = json.loads((self.root / "child.json").read_text())
        self.assertFalse(Path(info["cwd"]).exists())
        self.assertFalse(Path(info["env"]["HOME"]).exists())
        with self.assertRaises(ProcessLookupError):
            os.kill(info["pid"], 0)
        self.assertTrue(all(p.closed for p in FakeProxy.instances))
        self.assertFalse(any(t.name == "evobridge-app-server-reader" for t in threading.enumerate()))
        return info

    def test_native_handshake_isolation_and_frozen_configuration(self):
        runtime = self.runtime(logdir=self.root / "logs")
        with patch.dict(os.environ, {"PRIVATE_MODEL_KEY": "upstream-secret", "OPENAI_API_KEY": "other-secret",
                                     "CODEX_HOME": "/private/user/codex", "EVOBRIDGE_HOST_ONLY": "hidden"}):
            self.assertEqual(runtime.run("scope", "task", [TOOL], lambda *_: "unused"), "answer")
        info = self.assert_cleaned()
        self.assertNotEqual(info["env"]["HOME"], os.environ["HOME"])
        self.assertNotIn("PRIVATE_MODEL_KEY", info["env"])
        self.assertNotIn("OPENAI_API_KEY", info["env"])
        self.assertNotIn("EVOBRIDGE_HOST_ONLY", info["env"])
        self.assertEqual(info["env"]["EVOBRIDGE_PROXY_TOKEN"], "fixture-proxy-token")
        self.assertEqual(info["env"]["NO_PROXY"], "127.0.0.1,localhost,::1")
        self.assertIn("--strict-config", info["argv"])
        config = {v.split("=", 1)[0]: v.split("=", 1)[1] for i, v in enumerate(info["argv"])
                  if i > 0 and info["argv"][i - 1] == "-c"}
        self.assertEqual(config["model_context_window"], "131072")
        self.assertEqual(config["model_auto_compact_token_limit"], "98304")
        self.assertEqual(config["tool_output_token_limit"], "8192")
        self.assertEqual(config["model_reasoning_effort"], '"medium"')
        self.assertEqual(config["tools.experimental_request_user_input.enabled"], "false")
        self.assertEqual(config["cli_auth_credentials_store"], '"file"')
        self.assertEqual(config["mcp_oauth_credentials_store"], '"file"')
        for name in ("shell_tool", "unified_exec", "apps", "plugins", "hooks", "memories", "multi_agent",
                     "browser_use", "code_mode_host", "skill_search", "enable_request_compression",
                     "system_proxy_fallback"):
            self.assertEqual(config["features." + name], "false")
        self.assertEqual(config["features.skip_host_skill_discovery"], "true")
        trace = self.trace()
        self.assertTrue(trace[0]["params"]["capabilities"]["experimentalApi"])
        thread = next(v["params"] for v in trace if v.get("method") == "thread/start")
        self.assertEqual(thread["baseInstructions"], "scope")
        self.assertTrue(thread["ephemeral"])
        self.assertFalse(thread["allowProviderModelFallback"])
        self.assertEqual((thread["sandbox"], thread["approvalPolicy"]), ("read-only", "never"))
        self.assertEqual(thread["dynamicTools"], [{"type": "function", **TOOL}])
        self.assertEqual(FakeProxy.instances[0].allowed_tools, {"read"})
        turn = next(v["params"] for v in trace if v.get("method") == "turn/start")
        self.assertEqual(turn["input"], [{"type": "text", "text": "task"}])
        self.assertEqual(turn["effort"], "medium")
        for f in (self.root / "logs").rglob("*.log"):
            self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        self.assertNotIn("upstream-secret", json.dumps(info))
        self.assertEqual(runtime.session_receipts[0].status, "completed")

    def test_early_events_and_tool_request_are_not_lost(self):
        actions = [("emit", message("commentary", phase="commentary", ident="c")),
                   ("emit", call()), ("admit", None), ("receive", None),
                   ("emit", message("final")), ("emit", done())]
        dispatched = []
        runtime = self.runtime(actions)
        self.assertEqual(runtime.run("instructions", "prompt", [TOOL],
                                     lambda n, a: dispatched.append((n, a)) or {"contents": "ok"}), "final")
        self.assertEqual(dispatched, [("read", {"path": "memory.md"})])
        result = next(v["result"] for v in self.trace() if v.get("id") == 80)
        self.assertEqual(result, {"success": True, "contentItems": [{"type": "inputText", "text": '{"contents": "ok"}'}]})
        self.assert_cleaned()

    def test_completion_before_admission_response_is_buffered(self):
        runtime = self.runtime([("emit", message()), ("emit", done()), ("admit", None)])
        self.assertEqual(runtime.run("", "", [], lambda *_: None), "answer")
        self.assert_cleaned()

    def test_wrong_thread_turn_namespace_unknown_tools_and_approvals_denied(self):
        requests = [call(ident=80, thread="foreign"), call(ident=81, turn="foreign"),
                    call(ident=82, namespace="foreign"), call(ident=83, name="shell"),
                    call(ident=84, arguments="not an object"),
                    {"id": 85, "method": "item/commandExecution/requestApproval", "params": {}}]
        actions = [("admit", None)]
        for req in requests:
            actions.extend([("emit", req), ("receive", None)])
        actions.extend([("emit", message("foreign", thread="foreign")),
                        ("emit", done("failed", turn="foreign")),
                        ("emit", message()), ("emit", done())])
        runtime = self.runtime(actions)
        self.assertEqual(runtime.run("", "", [TOOL], lambda *_: self.fail("Ungrantable dispatch")), "answer")
        replies = [v for v in self.trace() if v.get("id") in range(80, 86)]
        self.assertEqual(len(replies), len(requests))
        self.assertTrue(all(v["error"]["code"] == -32601 for v in replies))
        self.assertEqual(runtime.session_receipts[0].denied_requests, 6)
        self.assert_cleaned()

    def test_duplicate_tool_calls_do_not_repeat_side_effects(self):
        actions = [("admit", None)]
        for req in [call(), call(ident=81), call(ident=82, arguments={"path": "different"})]:
            actions.extend([("emit", req), ("receive", None)])
        actions.extend([("emit", message()), ("emit", done())])
        runtime = self.runtime(actions)
        dispatched = []
        runtime.run("", "", [TOOL], lambda *args: dispatched.append(args) or "ok")
        self.assertEqual(len(dispatched), 1)
        replies = [v for v in self.trace() if v.get("id") in (80, 81, 82)]
        self.assertEqual(replies[0]["result"], replies[1]["result"])
        self.assertIn("error", replies[2])

    def test_tool_errors_are_reported_without_exception_secrets(self):
        def dispatch(*_):
            raise ValueError("a-secret-must-not-appear")
        actions = [("admit", None), ("emit", call()), ("receive", None),
                   ("emit", message()), ("emit", done())]
        self.runtime(actions).run("", "", [TOOL], dispatch)
        result = next(v["result"] for v in self.trace() if v.get("id") == 80)
        self.assertFalse(result["success"])
        self.assertNotIn("a-secret", json.dumps(result))

    def test_completed_final_only_and_authoritative_turn_messages(self):
        loaded = {"id": "loaded", "type": "agentMessage", "phase": "final_answer", "text": "authoritative"}
        actions = [("admit", None), ("emit", message("abandoned")),
                   ("emit", done(items=[loaded]))]
        self.assertEqual(self.runtime(actions).run("", "", [], lambda *_: None), "authoritative")
        for items in [[event("item/agentMessage/delta", itemId="partial", delta="partial final")],
                      [message("only commentary", phase="commentary")], []]:
            with self.subTest(items=items):
                runtime = self.runtime([("admit", None), *[("emit", i) for i in items], ("emit", done())])
                with self.assertRaisesRegex(CodexRuntimeError, "without a final"):
                    runtime.run("", "", [], lambda *_: None)
                self.assert_cleaned()

    def test_completed_unphased_message_compatibility(self):
        runtime = self.runtime([("admit", None), ("emit", message("legacy", phase=None)), ("emit", done())])
        self.assertEqual(runtime.run("", "", [], lambda *_: None), "legacy")
        self.assertEqual(runtime.session_receipts[0].final_message_source, "completed_unphased_agentMessage")

    def test_failed_and_incomplete_turns_never_return_partial_final(self):
        for status in ("failed", "interrupted", "incomplete"):
            with self.subTest(status=status):
                runtime = self.runtime([("admit", None), ("emit", message("partial")), ("emit", done(status))])
                with self.assertRaisesRegex(CodexRuntimeError, "did not complete"):
                    runtime.run("", "", [], lambda *_: None)
                self.assertEqual(runtime.session_receipts[0].status, "failed")
                self.assert_cleaned()

    def test_native_usage_compaction_and_provider_usage_are_distinct(self):
        usage = {"total": {"inputTokens": 100, "outputTokens": 40}, "last": {"inputTokens": 20, "outputTokens": 10}}
        actions = [("admit", None), ("emit", event("thread/tokenUsage/updated", tokenUsage=usage)),
                   ("emit", event("item/completed", item={"id": "compact", "type": "contextCompaction"})),
                   ("emit", event("thread/compacted")), ("emit", message()), ("emit", done())]
        FakeProxy.request_receipts = [ProviderReceipt(input_tokens=7), ProviderReceipt(input_tokens=11)]
        runtime = self.runtime(actions)
        runtime.run("", "", [], lambda *_: None)
        self.assertEqual(len(runtime.usage_receipts), 2)
        self.assertEqual(sum(v.input_tokens for v in runtime.usage_receipts), 18)
        self.assertEqual(runtime.session_receipts[0].native_token_usage, usage)
        self.assertEqual(runtime.session_receipts[0].compactions, 1)

    def test_failed_provider_terminal_rejected_and_receipt_preserved(self):
        FakeProxy.request_receipts = [ProviderReceipt(success=False, terminal_status="incomplete")]
        runtime = self.runtime()
        with self.assertRaisesRegex(CodexRuntimeError, "Provider request"):
            runtime.run("", "", [], lambda *_: None)
        self.assertEqual(len(runtime.usage_receipts), 1)
        self.assert_cleaned()

    def test_host_tool_lifecycle_fails_closed(self):
        runtime = self.runtime([("admit", None), ("emit", event("item/started", item={
            "id": "host", "type": "commandExecution", "command": "echo not allowed"}))])
        with self.assertRaisesRegex(CodexRuntimeError, "ungranted host tool"):
            runtime.run("", "", [], lambda *_: None)
        self.assert_cleaned()

    def test_startup_timeout_and_disconnect_cleanup(self):
        runtime = CodexRuntime("http://provider.invalid/v1", ModelSettings("model"),
                               executable=self.stub([], startup_delay=3), startup_timeout=1.5)
        with self.assertRaisesRegex(CodexRuntimeError, "timed out"):
            runtime.run("", "", [], lambda *_: None)
        self.assert_cleaned()
        runtime = self.runtime([("admit", None), ("exit", 3)])
        with self.assertRaisesRegex(CodexRuntimeError, "disconnected"):
            runtime.run("", "", [], lambda *_: None)
        self.assert_cleaned()

    def test_generation_has_no_startup_deadline(self):
        runtime = self.runtime([("admit", None), ("delay", 1.8), ("emit", message()), ("emit", done())],
                               startup_timeout=1.5)
        self.assertEqual(runtime.run("", "", [], lambda *_: None), "answer")
        self.assert_cleaned()

    def test_missing_executable_fails_without_fallback_and_closes_proxy(self):
        runtime = CodexRuntime("http://provider.invalid/v1", ModelSettings("model"), executable=self.root / "missing")
        with self.assertRaisesRegex(CodexRuntimeError, "Cannot launch"):
            runtime.run("", "", [], lambda *_: None)
        self.assertTrue(FakeProxy.instances[0].closed)

    def test_native_model_and_isolation_overrides_are_rejected(self):
        for override in ({"model": "different"}, {"modelProvider": "other"},
                         {"sandbox": {"type": "dangerFullAccess"}}, {"cwd": "/"},
                         {"reasoningEffort": "high"}):
            with self.subTest(override=override):
                runtime = CodexRuntime("http://provider.invalid/v1",
                    ModelSettings("model", reasoning_effort="medium"),
                    executable=self.stub([], thread_overrides=override))
                with self.assertRaisesRegex(CodexRuntimeError, "changed"):
                    runtime.run("", "", [], lambda *_: None)
                self.assert_cleaned()

    def test_fresh_home_on_every_run_and_cloud_empty_tools(self):
        runtime = self.runtime()
        runtime.run("", "", [], lambda *_: None)
        first = json.loads((self.root / "child.json").read_text())["env"]["HOME"]
        runtime.run("", "", [], lambda *_: None)
        second = json.loads((self.root / "child.json").read_text())["env"]["HOME"]
        self.assertNotEqual(first, second)
        self.assertEqual(FakeProxy.instances[1].allowed_tools, set())
        self.assertEqual(len(runtime.session_receipts), 2)
        self.assert_cleaned()

    def test_tool_validation_and_identity_omit_credentials(self):
        runtime = CodexRuntime("https://name:secret@example.com/v1?key=secret", ModelSettings("model"))
        self.assertEqual(runtime.runtime_identity["endpoint"], "https://example.com/v1")
        self.assertNotIn("secret", json.dumps(runtime.runtime_identity))
        for tools in ([TOOL, TOOL], [{"name": "x", "description": "x"}], [{**TOOL, "type": "namespace"}]):
            with self.assertRaises(ValueError):
                runtime.run("", "", tools, lambda *_: None)
        self.assertFalse(FakeProxy.instances)


if __name__ == "__main__":
    unittest.main()
