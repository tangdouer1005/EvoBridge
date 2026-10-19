"""Isolated native agent sessions with scoped dynamic tools.

Codex owns conversation history, tool scheduling, and context compaction. This
module only hosts its app-server protocol and forwards provider requests through
an authenticated, loopback Responses proxy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from .config import ModelSettings
from .responses import ResponsesProxy


class CodexRuntimeError(RuntimeError):
    """The native session failed; no alternative execution path was used."""


@dataclass
class SessionReceipt:
    """Native session metadata, separate from provider request accounting."""

    thread_id: str | None = None
    turn_id: str | None = None
    status: str = "starting"
    native_token_usage: dict | None = None
    compactions: int = 0
    denied_requests: int = 0
    final_message_source: str | None = None
    error: str | None = None


# These are actual feature names in the supported app-server. Disabling host
# integrations matters even with a read-only sandbox: reads are capabilities too.
_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "unified_exec_tty", "shell_snapshot",
    "shell_snapshot_v2", "view_image", "image_generation", "apps", "plugins",
    "remote_plugin", "recommended_plugins", "plugin_sharing", "multi_agent",
    "multi_agent_v2", "multi_agent_v2_dynamic_tools", "agent_message_board",
    "memories", "external_agent_memory_import", "hooks", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "browser_annotation_api",
    "computer_use", "in_app_browser", "code_mode", "code_mode_host",
    "code_mode_prewarm", "code_mode_only", "code_mode_tool_search",
    "workspace_dependencies", "skill_search", "skill_mcp_dependency_install",
    "sleep_tool", "goals", "tool_suggest", "default_mode_request_user_input",
    "send_message_to_user_async", "request_permissions_tool", "auth_elicitation",
    "tool_call_mcp_elicitation", "api_key_model_discovery", "daemon_auto_start",
    "unbounded_connection_retries", "rollout_budget", "standalone_web_search",
    "enable_request_compression", "respect_system_proxy", "system_proxy_fallback",
)
_FORBIDDEN_ITEMS = {
    "commandExecution", "fileChange", "mcpToolCall", "collabAgentToolCall",
    "subAgentActivity", "webSearch", "imageView", "imageGeneration", "sleep",
}


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + "=" + _toml(v) for k, v in value.items()) + "}"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _dynamic_tools(tools: list[dict]) -> list[dict]:
    result, names = [], set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ValueError("Only global function tools are supported")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Tool names must be nonempty and unique")
        if not isinstance(tool.get("description"), str) or not isinstance(tool.get("inputSchema"), dict):
            raise ValueError("Tools require a description and object inputSchema")
        names.add(name)
        # Do not pass unrecognized namespace/capability options through.
        result.append({"type": "function", "name": name,
                       "description": tool["description"], "inputSchema": tool["inputSchema"]})
    return result


class CodexRuntime:
    def __init__(self, base_url: str, settings: ModelSettings, api_key_env=None,
                 executable="codex", *, startup_timeout=30, logdir=None):
        if (isinstance(startup_timeout, bool) or not isinstance(startup_timeout, (int, float))
                or not math.isfinite(startup_timeout) or startup_timeout <= 0):
            raise ValueError("startup_timeout must be positive and finite")
        self.base_url, self.settings = base_url, settings
        self.api_key_env, self.executable = api_key_env, os.fspath(executable)
        self.startup_timeout = startup_timeout
        self.logdir = Path(logdir) if logdir is not None else None
        self.usage_receipts: list = []
        self.session_receipts: list[SessionReceipt] = []
        endpoint = urlsplit(base_url)
        host = endpoint.hostname or ""
        if ":" in host:
            host = "[" + host + "]"
        if endpoint.port is not None:
            host += ":" + str(endpoint.port)
        self.runtime_identity = {
            "backend": "codex-app-server", "protocol": "0.162.1",
            "endpoint": urlunsplit((endpoint.scheme, host, endpoint.path, "", "")),
            "executable": self.executable,
        }

    def _config(self, proxy_url: str) -> dict:
        result = {
            "model_provider": "evobridge", "model": self.settings.model,
            "model_context_window": self.settings.context_tokens,
            "model_auto_compact_token_limit": self.settings.compact_at,
            "model_auto_compact_token_limit_scope": "total",
            "tool_output_token_limit": self.settings.tool_output_tokens,
            "approval_policy": "never", "sandbox_mode": "read-only",
            "cli_auth_credentials_store": "file", "mcp_oauth_credentials_store": "file",
            "tools.experimental_request_user_input.enabled": False,
            "web_search": "disabled", "project_doc_max_bytes": 0,
            "project_doc_fallback_filenames": [], "analytics.enabled": False,
            "history.persistence": "none", "memories.generate_memories": False,
            "memories.use_memories": False, "features.skip_host_skill_discovery": True,
            "model_providers.evobridge": {
                "name": "EvoBridge", "base_url": proxy_url, "wire_api": "responses",
                "requires_openai_auth": False, "supports_websockets": False,
                "env_key": "EVOBRIDGE_PROXY_TOKEN", "request_max_retries": 0,
                "stream_max_retries": 0, "stream_idle_timeout_ms": 2147483647,
            },
            **{"features." + name: False for name in _DISABLED_FEATURES},
        }
        if self.settings.reasoning_effort is not None:
            result["model_reasoning_effort"] = self.settings.reasoning_effort
        return result

    def run(self, instructions: str, prompt: str, tools: list[dict],
            dispatch: Callable[[str, dict], Any]) -> str:
        dynamic = _dynamic_tools(tools)
        if not isinstance(instructions, str) or not isinstance(prompt, str):
            raise TypeError("Instructions and prompt must be strings")
        receipt = SessionReceipt()
        self.session_receipts.append(receipt)
        proxy = None
        try:
            proxy = ResponsesProxy(self.base_url, self.settings, self.api_key_env,
                                   allowed_tools={tool["name"] for tool in dynamic})
            with proxy:
                answer = self._session(proxy, receipt, instructions, prompt, dynamic, dispatch)
            # A native terminal cannot turn an incomplete provider generation
            # into an accepted answer. Accounting remains per actual request.
            for request in proxy.receipts:
                if (getattr(request, "attempted", False) and not getattr(request, "success", False)
                        or getattr(request, "terminal_status", None) in ("incomplete", "failed", "cancelled")):
                    raise CodexRuntimeError("Provider request did not complete successfully")
            receipt.status = "completed"
            return answer
        except BaseException as exc:
            receipt.status = "failed"
            receipt.error = type(exc).__name__
            raise
        finally:
            if proxy is not None:
                self.usage_receipts.extend(proxy.receipts)

    def _session(self, proxy, receipt, instructions, prompt, dynamic, dispatch):
        with tempfile.TemporaryDirectory(prefix="evobridge-agent-") as clean:
            root = Path(clean)
            home, cwd, codex_home = root / "home", root / "work", root / "home" / ".codex"
            home.mkdir(mode=0o700)
            cwd.mkdir(mode=0o700)
            codex_home.mkdir(mode=0o700)
            env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "LC_CTYPE")}
            env.update(HOME=str(home), CODEX_HOME=str(codex_home), TMPDIR=str(root),
                       XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
                       XDG_DATA_HOME=str(home / ".local" / "share"),
                       NO_PROXY="127.0.0.1,localhost,::1", no_proxy="127.0.0.1,localhost,::1",
                       EVOBRIDGE_PROXY_TOKEN=proxy.token)
            argv = [self.executable, "app-server", "--strict-config"]
            for key, value in self._config(proxy.base_url).items():
                argv.extend(("-c", key + "=" + _toml(value)))
            # Only explicit private logs retain native events or stderr. Neither
            # the command line nor these records contain the upstream API key.
            logroot = None
            if self.logdir is not None:
                self.logdir.mkdir(parents=True, exist_ok=True, mode=0o700)
                logroot = Path(tempfile.mkdtemp(prefix="session-", dir=self.logdir))
            stderr = self._private_log(logroot / "stderr.log") if logroot else open(os.devnull, "w")
            events = self._private_log(logroot / "events.jsonl") if logroot else None
            proc, reader = None, None
            messages: queue.Queue = queue.Queue()
            try:
                try:
                    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=stderr, cwd=cwd, env=env, text=True,
                                            encoding="utf-8", start_new_session=os.name == "posix")
                except OSError as exc:
                    raise CodexRuntimeError("Cannot launch the required Codex app-server executable") from exc

                def pump():
                    try:
                        for line in proc.stdout:
                            messages.put(line)
                    except (OSError, UnicodeError) as exc:
                        messages.put(exc)
                    finally:
                        messages.put(None)

                reader = threading.Thread(target=pump, name="evobridge-app-server-reader", daemon=True)
                reader.start()
                return self._protocol(proc, messages, events, receipt, instructions, prompt,
                                      dynamic, dispatch, str(cwd))
            finally:
                try:
                    if proc is not None:
                        self._stop(proc, reader)
                finally:
                    stderr.close()
                    if events is not None:
                        events.close()

    @staticmethod
    def _private_log(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        return os.fdopen(fd, "w", encoding="utf-8")

    @staticmethod
    def _stop(proc, reader):
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        # Stop only this invocation's process group, including any descendants
        # holding its stdout pipe open. Never signal another user's Codex.
        if os.name == "posix" and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                if proc.poll() is None:
                    proc.terminate()
        elif proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    if proc.poll() is None:
                        proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=2)
        if reader is not None:
            reader.join(timeout=2)
            if reader.is_alive() and os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                reader.join(timeout=2)
        if proc.stdout is not None:
            proc.stdout.close()

    def _protocol(self, proc, messages, events, receipt, instructions, prompt, dynamic, dispatch, cwd):
        pending = deque()
        deadline = time.monotonic() + self.startup_timeout
        serial = 0
        granted = {tool["name"] for tool in dynamic}
        calls, agent_messages, compacted = {}, {}, set()
        legacy_compactions = 0

        def send(value):
            try:
                proc.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise CodexRuntimeError("Codex app-server disconnected") from exc

        def receive(starting=False):
            remaining = deadline - time.monotonic() if starting else None
            if remaining is not None and remaining <= 0:
                raise CodexRuntimeError("Codex app-server startup/RPC timed out")
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty:
                raise CodexRuntimeError("Codex app-server startup/RPC timed out") from None
            if line is None or isinstance(line, BaseException):
                raise CodexRuntimeError("Codex app-server disconnected")
            try:
                msg = json.loads(line)
            except (ValueError, TypeError):
                raise CodexRuntimeError("Invalid Codex app-server message") from None
            if not isinstance(msg, dict):
                raise CodexRuntimeError("Invalid Codex app-server message")
            if events is not None:
                events.write(json.dumps(msg, ensure_ascii=False) + "\n")
                events.flush()
            return msg

        def matches(params):
            return (receipt.thread_id is not None and receipt.turn_id is not None
                    and params.get("threadId") == receipt.thread_id
                    and params.get("turnId", (params.get("turn") or {}).get("id")) == receipt.turn_id)

        def deny(msg):
            receipt.denied_requests += 1
            send({"id": msg["id"], "error": {"code": -32601, "message": "Capability not granted"}})

        def request(msg):
            params = msg.get("params") or {}
            if (msg["method"] != "item/tool/call" or not matches(params)
                    or params.get("namespace") is not None or params.get("tool") not in granted
                    or not isinstance(params.get("arguments"), dict)
                    or not isinstance(params.get("callId"), str)):
                deny(msg)
                return
            call_id = params["callId"]
            signature = json.dumps([params["tool"], params["arguments"]], sort_keys=True)
            if call_id in calls:
                previous_signature, result = calls[call_id]
                if signature != previous_signature:
                    deny(msg)
                    return
            else:
                try:
                    value = dispatch(params["tool"], params["arguments"])
                    output = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
                    result = {"success": True, "contentItems": [{"type": "inputText", "text": output}]}
                except Exception as exc:
                    result = {"success": False, "contentItems": [{"type": "inputText",
                              "text": "Tool dispatch failed: " + type(exc).__name__}]}
                calls[call_id] = signature, result
            send({"id": msg["id"], "result": result})

        def rpc(method, params):
            nonlocal serial
            serial += 1
            ident = "evobridge-" + str(serial)
            send({"id": ident, "method": method, "params": params})
            while True:
                msg = receive(starting=True)
                if "method" in msg:
                    if "id" in msg and not (method == "turn/start" and msg["method"] == "item/tool/call"):
                        request(msg)
                    else:
                        pending.append(msg)
                elif msg.get("id") == ident:
                    if "error" in msg or not isinstance(msg.get("result"), dict):
                        raise CodexRuntimeError("Codex app-server RPC failed: " + method)
                    return msg["result"]

        rpc("initialize", {"clientInfo": {"name": "evobridge", "version": "0.1.0"},
                           "capabilities": {"experimentalApi": True}})
        send({"method": "initialized", "params": {}})
        result = rpc("thread/start", {"model": self.settings.model, "modelProvider": "evobridge",
            "allowProviderModelFallback": False, "cwd": cwd, "ephemeral": True,
            "sandbox": "read-only", "approvalPolicy": "never", "dynamicTools": dynamic,
            "baseInstructions": instructions, "developerInstructions": "",
            "environments": [], "runtimeWorkspaceRoots": [], "selectedCapabilityRoots": []})
        if result.get("model") != self.settings.model or result.get("modelProvider") != "evobridge":
            raise CodexRuntimeError("Codex changed the configured model or provider")
        if result.get("approvalPolicy") != "never" or (result.get("sandbox") or {}).get("type") != "readOnly":
            raise CodexRuntimeError("Codex changed the required isolation policy")
        if Path(result.get("cwd", "")).resolve() != Path(cwd).resolve():
            raise CodexRuntimeError("Codex changed the isolated working directory")
        if (self.settings.reasoning_effort is not None
                and result.get("reasoningEffort") != self.settings.reasoning_effort):
            raise CodexRuntimeError("Codex changed the configured reasoning effort")
        receipt.thread_id = (result.get("thread") or {}).get("id")
        if not isinstance(receipt.thread_id, str) or not receipt.thread_id:
            raise CodexRuntimeError("Codex did not create a thread")
        params = {"threadId": receipt.thread_id, "input": [{"type": "text", "text": prompt}]}
        if self.settings.reasoning_effort is not None:
            params["effort"] = self.settings.reasoning_effort
        result = rpc("turn/start", params)
        receipt.turn_id = (result.get("turn") or {}).get("id")
        if not isinstance(receipt.turn_id, str) or not receipt.turn_id:
            raise CodexRuntimeError("Codex did not admit a turn")
        receipt.status = "running"

        def completed_item(item):
            kind = item.get("type")
            if kind in _FORBIDDEN_ITEMS:
                raise CodexRuntimeError("Codex emitted an ungranted host tool")
            if kind == "agentMessage" and isinstance(item.get("id"), str):
                if isinstance(item.get("text"), str):
                    agent_messages[item["id"]] = item
            elif kind == "contextCompaction" and isinstance(item.get("id"), str):
                compacted.add(item["id"])

        # No generation deadline: after admission Codex owns the full agent loop.
        while True:
            msg = pending.popleft() if pending else receive()
            if "method" in msg and "id" in msg:
                request(msg)
                continue
            params = msg.get("params") or {}
            if not matches(params):
                continue
            method = msg.get("method")
            if method == "item/started" and (params.get("item") or {}).get("type") in _FORBIDDEN_ITEMS:
                raise CodexRuntimeError("Codex emitted an ungranted host tool")
            if method == "item/completed":
                completed_item(params.get("item") or {})
            elif method == "thread/tokenUsage/updated":
                receipt.native_token_usage = params.get("tokenUsage")
            elif method == "thread/compacted":
                legacy_compactions += 1
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                if turn.get("status") != "completed" or turn.get("error") is not None:
                    raise CodexRuntimeError("Codex turn did not complete successfully")
                # Loaded completed-turn messages supersede streamed items (for
                # example when a reconnect abandoned an earlier final prefix).
                loaded = [item for item in turn.get("items", []) if item.get("type") == "agentMessage"]
                if loaded:
                    agent_messages.clear()
                for item in turn.get("items", []):
                    completed_item(item)
                receipt.compactions = max(len(compacted), legacy_compactions)
                break
            receipt.compactions = max(len(compacted), legacy_compactions)

        rows = list(agent_messages.values())
        final = [item["text"] for item in rows if item.get("phase") == "final_answer"]
        if final:
            answer = "\n\n".join(final)
            receipt.final_message_source = "final_answer"
        elif rows and all(item.get("phase") is None for item in rows):
            answer = rows[-1]["text"]
            receipt.final_message_source = "completed_unphased_agentMessage"
        else:
            raise CodexRuntimeError("Codex completed without a final agent message")
        if not answer.strip():
            raise CodexRuntimeError("Codex completed with an empty final agent message")
        return answer
