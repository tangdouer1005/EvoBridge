"""A loopback Responses transport for native Codex sessions.

Codex owns tools, its agent loop, and compaction. This relay only pins the
generation profile, supplies the explicitly configured provider credential, and
observes provider accounting. It never retries or follows redirects. Provider
response bytes (including errors and SSE) are forwarded without modification.

The configured endpoint must implement the Responses API, including the sampling
extensions used by the local model. A Chat Completions endpoint is not an adapter.
``thinking=False`` pins ``reasoning.effort`` to ``none``; a Qwen gateway must map
that setting to its model's thinking control. No chat-only template fields are
invented. The compact endpoint retains its narrower native request schema.
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass
import http.client
import http.server
import json
import math
import os
import secrets
import socket
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .config import ModelSettings


@dataclass
class RequestReceipt:
    """One actual upstream request; unknown provider counters stay ``None``.

    Receipts are appended before attempting HTTP and settled before the proxy
    context exits. ``success`` means a complete provider response, not that a
    native session or its task succeeded. Incomplete/failed terminals still have
    billable usage and therefore retain any supplied counters.
    """

    model: str
    path: str
    attempted: bool = True
    success: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    elapsed_seconds: float = 0.0
    terminal_status: str | None = None
    http_status: int | None = None
    request_id: str | None = None
    error: str | None = None


_COUNTERS = {
    ("usage", "input_tokens"): "input_tokens",
    ("usage", "output_tokens"): "output_tokens",
    ("usage", "total_tokens"): "total_tokens",
    ("usage", "input_tokens_details", "cached_tokens"): "cached_tokens",
    ("usage", "output_tokens_details", "reasoning_tokens"): "reasoning_tokens",
}
_FIELDS = {("status",): "status", ("id",): "id", ("type",): "type",
           ("object",): "object", **_COUNTERS}
_FIELDS.update({("response",) + key: value for key, value in tuple(_FIELDS.items())
                if key != ("type",)})
_TERMINALS = {"response.completed": "completed", "response.incomplete": "incomplete",
              "response.failed": "failed", "error": "failed"}


class _AccountingJSON:
    """Small selective JSON reader with bounded scalar storage.

    Full Responses terminals repeat generated output and can be arbitrarily
    large. A normal json.loads buffer would retain that output just to read a
    handful of counters. This reader tracks JSON structure but only stores short
    scalar tokens and the accounting paths above. It is observational: malformed
    accounting never changes the bytes sent to the native client.
    """

    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.stack: list[dict[str, Any]] = []
        self.fields: dict[str, Any] = {}
        self.mode: str | None = None
        self.token = ""
        self.large = False
        self.escaped = False
        self.invalid = False
        self.complete = False
        self.started = False

    def _remember(self, text: str):
        if not self.large:
            if len(self.token) + len(text) <= 256:
                self.token += text
            else:
                self.token = ""
                self.large = True

    def feed(self, data: bytes):
        if self.invalid:
            return
        try:
            self._characters(self.decoder.decode(data))
        except (UnicodeError, ValueError, TypeError, IndexError):
            self.invalid = True

    def finish(self) -> dict[str, Any]:
        if not self.invalid:
            try:
                self._characters(self.decoder.decode(b"", final=True))
                if self.mode == "literal":
                    self._literal()
                if self.mode or self.stack or not self.complete:
                    self.invalid = True
            except (UnicodeError, ValueError, TypeError, IndexError):
                self.invalid = True
        return {} if self.invalid else self.fields

    def _characters(self, text: str):
        for char in text:
            if self.mode == "string":
                if char == '"' and not self.escaped:
                    value = None if self.large else json.loads('"' + self.token + '"')
                    self.mode = None
                    self._scalar(value, string=True)
                else:
                    self._remember(char)
                    if self.escaped:
                        self.escaped = False
                    elif char == "\\":
                        self.escaped = True
                continue
            if self.mode == "literal":
                if char not in " \r\n\t,]}":
                    self._remember(char)
                    continue
                self._literal()
            if char in " \r\n\t":
                continue
            if char == '"':
                self.mode, self.token, self.large, self.escaped = "string", "", False, False
            elif char in "{[":
                path = self._value_path()
                self._consume_value()
                if len(self.stack) >= 128:
                    raise ValueError("Accounting nesting limit")
                if path in (("error",), ("response", "error")):
                    self.fields["has_error"] = True
                self.stack.append({"kind": char, "path": path,
                                   "state": "key" if char == "{" else "value", "key": None})
            elif char in "}]":
                frame = self.stack.pop()
                if (frame["kind"], char) not in (("{", "}"), ("[", "]")):
                    raise ValueError("Mismatched accounting delimiter")
                if frame["state"] not in ("key", "value", "next"):
                    raise ValueError("Incomplete accounting object")
                if not self.stack:
                    self.complete = True
            elif char == ":":
                if self.stack[-1]["state"] != "colon":
                    raise ValueError("Unexpected accounting colon")
                self.stack[-1]["state"] = "value"
            elif char == ",":
                if self.stack[-1]["state"] != "next":
                    raise ValueError("Unexpected accounting comma")
                self.stack[-1]["state"] = "key" if self.stack[-1]["kind"] == "{" else "value"
            else:
                self.mode, self.token, self.large = "literal", "", False
                self._remember(char)

    def _literal(self):
        if self.large:
            raise ValueError("Invalid accounting scalar")
        value = json.loads(self.token)
        self.mode = None
        self._scalar(value, string=False)

    def _value_path(self):
        if not self.stack:
            if self.started:
                raise ValueError("Multiple accounting values")
            return ()
        frame = self.stack[-1]
        if frame["state"] != "value":
            raise ValueError("Unexpected accounting value")
        return frame["path"] + ((frame["key"] if frame["kind"] == "{" else None),)

    def _consume_value(self):
        if self.stack:
            self.stack[-1]["state"] = "next"
        else:
            self.started = True

    def _scalar(self, value: Any, *, string: bool):
        if self.stack and self.stack[-1]["state"] == "key":
            if not string:
                raise ValueError("Accounting key is not a string")
            self.stack[-1].update(key=value, state="colon")
            return
        path = self._value_path()
        self._consume_value()
        if path in _FIELDS:
            self.fields[_FIELDS[path]] = value
        if path in (("error",), ("response", "error")) and value:
            self.fields["has_error"] = True
        if not self.stack:
            self.complete = True


class _Accounting:
    """Incremental SSE/JSON accounting; even a single huge SSE line is bounded."""

    def __init__(self, content_type: str):
        self.sse = "text/event-stream" in content_type.lower()
        self.parser = _AccountingJSON()
        self.fields: dict[str, Any] = {}
        self.prefix = b""
        self.mode: str | None = None
        self.line_length = 0
        self.line_has_non_cr = False
        self.data_started = False
        self.has_data = False
        self.event_name = b""

    def feed(self, block: bytes):
        if not self.sse:
            self.parser.feed(block)
            return
        parts = block.split(b"\n")
        for index, part in enumerate(parts):
            self._fragment(part)
            if index != len(parts) - 1:
                self._end_line()

    def _fragment(self, part: bytes):
        self.line_length += len(part)
        self.line_has_non_cr = self.line_has_non_cr or bool(part.strip(b"\r"))
        if self.mode is None:
            count = min(max(0, 6 - len(self.prefix)), len(part))
            self.prefix += part[:count]
            part = part[count:]
            if self.prefix.startswith(b"data:"):
                self.mode = "data"
                part = self.prefix[5:] + part
            elif self.prefix.startswith(b"event:"):
                self.mode = "event"
                part = self.prefix[6:] + part
            elif not (b"data:".startswith(self.prefix) or b"event:".startswith(self.prefix)):
                self.mode = "ignore"
            else:
                return
        if self.mode == "data":
            if not self.data_started and part:
                part = part[1:] if part.startswith(b" ") else part
                self.data_started = True
            self.parser.feed(part)
            self.has_data = True
        elif self.mode == "event" and len(self.event_name) < 128:
            self.event_name += part[:128 - len(self.event_name)]

    def _end_line(self):
        if not self.line_has_non_cr:
            self._end_event()
        elif self.mode == "data":
            self.parser.feed(b"\n")
        self.prefix, self.mode, self.line_length = b"", None, 0
        self.line_has_non_cr = False
        self.data_started = False

    def _end_event(self):
        if self.has_data:
            fields = self.parser.finish()
            event = fields.get("type") or self.event_name.strip().decode("ascii", errors="replace")
            if event in _TERMINALS:
                fields["status"] = _TERMINALS[event]
                self.fields = fields
        self.parser = _AccountingJSON()
        self.has_data = False
        self.event_name = b""

    def finish(self) -> dict[str, Any]:
        if not self.sse:
            return self.parser.finish()
        if self.line_length:
            self._end_line()
        if self.has_data:
            self._end_event()
        return self.fields


_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade"}
_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
_ROUTES = {"/v1/responses", "/v1/responses/compact"}


def _shutdown(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


class _ProxyServer(http.server.ThreadingHTTPServer):
    """Own handler lifetimes without an unbounded join on exceptional exit."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, proxy):
        self.proxy = proxy
        self._workers: dict[threading.Thread, socket.socket] = {}
        self._workers_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _Handler)

    def process_request(self, request, client_address):
        worker = threading.Thread(target=self.process_request_thread,
                                  args=(request, client_address), daemon=True)
        with self._workers_lock:
            self._workers[worker] = request
        worker.start()

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._workers_lock:
                self._workers.pop(threading.current_thread(), None)

    def finish_requests(self, *, cancelled: bool):
        with self._workers_lock:
            workers = list(self._workers.items())
        if cancelled:
            for _, connection in workers:
                _shutdown(connection)
        deadline = time.monotonic() + 1 if cancelled else None
        for worker, _ in workers:
            worker.join(None if deadline is None else max(0, deadline - time.monotonic()))


class ResponsesProxy:
    """Serve an authenticated ephemeral loopback endpoint inside a context.

    Configure Codex with ``base_url=proxy.base_url``, ``wire_api="responses"``,
    ``requires_openai_auth=false``, no retries, and an ``env_key`` containing
    ``proxy.token``. That token authenticates only to this relay. The upstream
    bearer credential comes exclusively from ``api_key_env``, if provided.

    ``connection_timeout`` bounds connection establishment only. Waiting for
    provider headers and generation has no deadline. Context
    normal exit waits for existing requests to finish, including after client
    disconnect. Exceptional exit closes owned sockets and joins for at most one
    second, marking interrupted requests failed without replaying them.
    """

    def __init__(self, base_url: str, settings: ModelSettings,
                 api_key_env: str | None = None, *, connection_timeout: float = 30,
                 allowed_tools: set[str] | None = None, header_timeout: float | None = None):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Expected an HTTP(S) Responses API root without credentials/query/fragment")
        # Fail early for malformed/out-of-range ports rather than recording a
        # local configuration error as an upstream request.
        parsed.port
        if header_timeout is not None:
            connection_timeout = header_timeout
        if (type(connection_timeout) not in (int, float) or not math.isfinite(connection_timeout)
                or connection_timeout <= 0):
            raise ValueError("connection_timeout must be positive and finite")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise ValueError("api_key_env must name an environment variable")
        if allowed_tools is not None and (not isinstance(allowed_tools, (set, frozenset))
                                           or any(not isinstance(name, str) or not name for name in allowed_tools)):
            raise ValueError("allowed_tools must be a set of nonempty function names")
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions") or path.endswith("/responses/compact"):
            raise ValueError("Configure a Responses API root, not a chat/compact endpoint")
        self._upstream = parsed
        self._upstream_path = path[:-len("/responses")] if path.endswith("/responses") else path
        self.settings = settings
        self.api_key_env = api_key_env
        self.connection_timeout = connection_timeout
        self.allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else None
        self.token = secrets.token_urlsafe(32)
        self.receipts: list[RequestReceipt] = []
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._connections: dict[http.client.HTTPConnection, socket.socket | None] = {}
        self._server: _ProxyServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("ResponsesProxy is not running")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self) -> ResponsesProxy:
        if self._server is not None:
            raise RuntimeError("ResponsesProxy is already running")
        if self._connections:
            raise RuntimeError("A cancelled proxy request is still shutting down")
        self._cancelled.clear()
        server = _ProxyServer(self)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        if self._server is not None:
            self._server.shutdown()
            if exc_type is not None:
                self._cancelled.set()
                with self._lock:
                    connections = list(self._connections.items())
                for conn, sock in connections:
                    _shutdown(sock or conn.sock)
            self._server.server_close()
            self._server.finish_requests(cancelled=exc_type is not None)
            self._thread.join()
            self._server = None
            self._thread = None

    def _payload(self, raw: bytes, path: str) -> bytes:
        def invalid_constant(_):
            raise ValueError("Non-finite JSON value")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        body = json.loads(raw, parse_constant=invalid_constant, object_pairs_hook=unique)
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON request object")
        if "model" in body and body["model"] != self.settings.model:
            raise ValueError("Request model differs from the configured model")
        body["model"] = self.settings.model
        if path == "/v1/responses":
            if self.allowed_tools is not None:
                offerings = body.get("tools", [])
                if not isinstance(offerings, list) or any(
                        not isinstance(tool, dict) or tool.get("type") != "function"
                        or tool.get("name") not in self.allowed_tools for tool in offerings):
                    raise ValueError("Request offers a tool not granted by this runtime")
            for name in _SAMPLING:
                value = getattr(self.settings, name)
                if value is None:
                    body.pop(name, None)
                else:
                    body[name] = value
            body.update(max_output_tokens=self.settings.max_output_tokens, store=False)
            effort = "none" if self.settings.thinking is False else self.settings.reasoning_effort
            if effort is not None:
                reasoning = body.get("reasoning") or {}
                if not isinstance(reasoning, dict):
                    raise ValueError("reasoning must be a JSON object")
                body["reasoning"] = {**reasoning, "effort": effort}
        return json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *_):
        pass

    def _reject(self, status: int, message: str):
        data = json.dumps({"error": {"message": message}}).encode()
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def do_GET(self):
        self._reject(405, "Only POST is supported")

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_GET

    def do_POST(self):
        proxy = self.server.proxy
        if self.path not in _ROUTES:
            self._reject(404, "Unsupported Responses API route")
            return
        authorization = self.headers.get_all("Authorization", [])
        if len(authorization) != 1 or not secrets.compare_digest(
                authorization[0].encode("utf-8"), ("Bearer " + proxy.token).encode("utf-8")):
            self._reject(401, "Invalid loopback credential")
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise ValueError("One Content-Length and no Transfer-Encoding is required")
            length = int(lengths[0])
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError("Request size is outside the supported range")
            if self.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Compressed requests are unsupported")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            payload = proxy._payload(raw, self.path)
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json",
                       "Accept-Encoding": "identity"}
            if proxy.api_key_env:
                credential = os.environ.get(proxy.api_key_env)
                if not credential or "\r" in credential or "\n" in credential:
                    raise ValueError("Configured API credential environment variable is empty or invalid")
                headers["Authorization"] = "Bearer " + credential
        except (ValueError, TypeError, UnicodeError, OSError, RecursionError):
            self._reject(400, "Invalid request or missing configured provider credential")
            return

        self.connection.settimeout(None)
        connection_type = http.client.HTTPSConnection if proxy._upstream.scheme == "https" else http.client.HTTPConnection
        conn = connection_type(proxy._upstream.hostname, proxy._upstream.port, timeout=proxy.connection_timeout)
        receipt = RequestReceipt(model=proxy.settings.model, path=self.path)
        started = time.monotonic()
        with proxy._lock:
            proxy.receipts.append(receipt)
            proxy._connections[conn] = None
        sent_headers = False
        observer = None
        read_complete = False
        try:
            # http.client does not inherit proxy settings, desktop auth, cookies,
            # or redirect/retry handlers from the host environment.
            upstream_path = proxy._upstream_path + self.path[len("/v1"):]
            conn.connect()
            with proxy._lock:
                proxy._connections[conn] = conn.sock
            if proxy._cancelled.is_set():
                raise OSError("Proxy cancelled")
            conn.sock.settimeout(None)
            conn.request("POST", upstream_path, body=payload, headers=headers)
            upstream = conn.getresponse()
            receipt.http_status = upstream.status
            sock = conn.sock or getattr(getattr(getattr(upstream, "fp", None), "raw", None), "_sock", None)
            if sock is not None:
                sock.settimeout(None)
            encoding = upstream.getheader("Content-Encoding", "identity").lower()
            if encoding == "identity":
                observer = _Accounting(upstream.getheader("Content-Type", ""))
            else:
                receipt.error = "UnsupportedAccountingEncoding"
            connected = True
            try:
                self.send_response_only(upstream.status, upstream.reason)
                connection_headers = {x.strip().lower() for x in upstream.getheader("Connection", "").split(",")}
                for name, value in upstream.getheaders():
                    # A native HTTP client may follow Location itself, bypassing
                    # this endpoint's credential, tool, and accounting boundary.
                    # Preserve the upstream status/body, but never redirect it.
                    if 300 <= upstream.status < 400 and name.lower() == "location":
                        continue
                    if name.lower() not in _HOP_HEADERS | connection_headers:
                        self.send_header(name, value)
                self.send_header("Connection", "close")
                self.end_headers()
            except OSError:
                connected = False
            sent_headers = True
            while True:
                try:
                    block = upstream.read1(65536)
                except http.client.IncompleteRead as exc:
                    block = exc.partial
                    if observer:
                        observer.feed(block)
                    if connected:
                        try:
                            self.wfile.write(block)
                            self.wfile.flush()
                        except OSError:
                            pass
                    raise
                if not block:
                    read_complete = True
                    break
                if observer:
                    observer.feed(block)
                if connected:
                    try:
                        self.wfile.write(block)
                        self.wfile.flush()
                    except OSError:
                        connected = False
            # A premature fixed-length EOF is not necessarily raised by read1.
            if upstream.length not in (None, 0):
                read_complete = False
                receipt.error = "IncompleteRead"
        except (OSError, http.client.HTTPException) as exc:
            receipt.error = "Cancelled" if proxy._cancelled.is_set() else type(exc).__name__
            if not sent_headers:
                self._reject(502, "Upstream Responses request failed")
        finally:
            if proxy._cancelled.is_set():
                receipt.error = "Cancelled"
            if observer is not None:
                fields = observer.finish()
                status = fields.get("status")
                receipt.terminal_status = status if isinstance(status, str) else None
                ident = fields.get("id")
                receipt.request_id = ident if isinstance(ident, str) else None
                for name in _COUNTERS.values():
                    value = fields.get(name)
                    setattr(receipt, name, value if type(value) is int and value >= 0 else None)
                if fields.get("has_error") and receipt.error is None:
                    receipt.error = "ProviderError"
                compact = self.path.endswith("/compact") and fields.get("object") == "response.compaction"
                receipt.success = (read_complete and receipt.error is None and
                                   receipt.http_status is not None and 200 <= receipt.http_status < 300 and
                                   (receipt.terminal_status == "completed" or compact))
                if compact and receipt.terminal_status is None:
                    receipt.terminal_status = "completed"
                if receipt.error is None and not receipt.success:
                    receipt.error = ("HTTPError" if receipt.http_status is not None and receipt.http_status >= 300
                                     else "ProviderIncomplete" if receipt.terminal_status == "incomplete"
                                     else "ProviderFailed" if receipt.terminal_status == "failed"
                                     else "MissingTerminal")
            receipt.elapsed_seconds = time.monotonic() - started
            conn.close()
            with proxy._lock:
                proxy._connections.pop(conn, None)
            self.close_connection = True
