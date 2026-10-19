"""Real loopback HTTP contracts; no external provider or model is required."""
from contextlib import contextmanager
import gzip
import http.client
import http.server
import json
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from evobridge.config import ModelSettings
from evobridge.responses import ResponsesProxy, _Accounting


def terminal(status="completed", usage=None, **extra):
    return {"id": "response-test", "object": "response", "status": status,
            "output": [], "usage": usage, **extra}


def sse(value, event=None):
    return ((f"event: {event}\n" if event else "") + "data: " +
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()


def response_writer(body, status=200, content_type="application/json", headers=None,
                    header_delay=0, body_delay=0):
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()

    def write(handler):
        time.sleep(header_delay)
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.flush()
        time.sleep(body_delay)
        handler.wfile.write(body)
        handler.wfile.flush()

    return write


@contextmanager
def upstream(*writers):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, dict(self.headers), json.loads(raw)))
            writers[len(calls) - 1](self)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def open_response(proxy, body=None, path="/v1/responses", headers=None):
    address = urlsplit(proxy.base_url)
    conn = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
    outgoing = {"Content-Type": "application/json", "Authorization": "Bearer " + proxy.token}
    outgoing.update(headers or {})
    raw = body if isinstance(body, bytes) else json.dumps(body or {"input": "hello"}).encode()
    conn.request("POST", path, body=raw, headers=outgoing)
    return conn, conn.getresponse()


def request(proxy, body=None, path="/v1/responses", headers=None):
    conn, response = open_response(proxy, body, path, headers)
    try:
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


class ResponsesProxyTests(unittest.TestCase):
    def test_sse_forwarding_pins_profile_and_keeps_tool_schema(self):
        usage = {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24,
                 "input_tokens_details": {"cached_tokens": 7},
                 "output_tokens_details": {"reasoning_tokens": 0}}
        wire = (b": provider heartbeat\r\n\r\n" +
                sse({"type": "response.output_text.delta", "delta": "原样输出"}) +
                sse({"type": "response.completed", "response": terminal(usage=usage)}) +
                b"data: [DONE]\n\n")
        tools = [{"type": "function", "name": "read_file", "description": "Read exact text",
                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                 "required": ["path"], "additionalProperties": False}, "strict": True}]
        body = {"input": [{"role": "user", "content": "private input"}], "tools": tools,
                "model": "Qwen3.5-9B", "stream": True, "temperature": 9, "top_p": .1,
                "top_k": 100, "min_p": .5, "presence_penalty": 2, "repetition_penalty": 3,
                "max_output_tokens": 1, "store": True, "reasoning": {"effort": "high", "summary": "auto"}}
        with upstream(response_writer(wire, content_type="text/event-stream")) as (url, calls):
            proxy = ResponsesProxy(url, ModelSettings.local(), allowed_tools={"read_file"})
            with proxy:
                status, headers, received = request(proxy, body)
            self.assertEqual((status, received), (200, wire))
            self.assertEqual(headers["Content-Type"], "text/event-stream")
            self.assertEqual(len(calls), 1)
            path, _, sent = calls[0]
            self.assertEqual(path, "/v1/responses")
            for name, value in {"temperature": .6, "top_p": .95, "top_k": 20, "min_p": 0,
                                "presence_penalty": 0, "repetition_penalty": 1.,
                                "max_output_tokens": 32768, "store": False}.items():
                self.assertEqual(sent[name], value)
            self.assertEqual(sent["reasoning"], {"effort": "none", "summary": "auto"})
            self.assertEqual(sent["tools"], tools)
            self.assertEqual(sent["input"], body["input"])
            self.assertNotIn("chat_template_kwargs", sent)
            receipt = proxy.receipts[0]
            self.assertTrue(receipt.attempted)
            self.assertTrue(receipt.success)
            self.assertEqual((receipt.input_tokens, receipt.output_tokens, receipt.total_tokens), (20, 4, 24))
            self.assertEqual((receipt.cached_tokens, receipt.reasoning_tokens), (7, 0))
            self.assertEqual(receipt.terminal_status, "completed")
            self.assertEqual(receipt.request_id, "response-test")
            self.assertGreaterEqual(receipt.elapsed_seconds, 0)

    def test_only_explicit_env_credential_reaches_provider(self):
        with upstream(response_writer(terminal()), response_writer(terminal())) as (url, calls):
            with patch.dict(os.environ, {"EVOBRIDGE_TEST_KEY": "explicit-key", "OPENAI_API_KEY": "desktop-key",
                                         "HTTP_PROXY": "http://localhost:1", "HTTPS_PROXY": "http://localhost:1"}):
                with ResponsesProxy(url, ModelSettings("cloud", reasoning_effort="medium"),
                                    "EVOBRIDGE_TEST_KEY", allowed_tools=set()) as proxy:
                    status, _, _ = request(proxy, {"input": "public", "temperature": 1,
                                                  "reasoning": {"effort": "low"}},
                                           headers={"Cookie": "desktop-cookie", "OpenAI-Organization": "desktop-org"})
                    self.assertEqual(status, 200)
                with ResponsesProxy(url, ModelSettings("cloud")) as proxy:
                    self.assertEqual(request(proxy)[0], 200)
            first_headers, first_body = calls[0][1:]
            self.assertEqual(first_headers["Authorization"], "Bearer explicit-key")
            self.assertNotIn("Cookie", first_headers)
            self.assertNotIn("OpenAI-Organization", first_headers)
            self.assertEqual(first_headers["Accept-Encoding"], "identity")
            self.assertEqual(first_body["reasoning"], {"effort": "medium"})
            self.assertNotIn("temperature", first_body)
            self.assertNotIn("Authorization", calls[1][1])

    def test_compact_route_keeps_native_narrow_schema(self):
        result = {"id": "compact-1", "object": "response.compaction", "output": [],
                  "usage": {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112}}
        body = {"model": "Qwen3.5-9B", "input": [], "instructions": "Preserve task constraints"}
        with upstream(response_writer(result)) as (url, calls):
            proxy = ResponsesProxy(url + "/responses", ModelSettings.local())
            with proxy:
                self.assertEqual(request(proxy, body, "/v1/responses/compact")[0], 200)
            self.assertEqual(calls[0][0], "/v1/responses/compact")
            self.assertEqual(calls[0][2], body)
            self.assertTrue(proxy.receipts[0].success)
            self.assertEqual(proxy.receipts[0].path, "/v1/responses/compact")
            self.assertEqual(proxy.receipts[0].input_tokens, 100)

    def test_failure_and_incomplete_sse_preserve_usage(self):
        for status in ("failed", "incomplete"):
            with self.subTest(status=status):
                result = terminal(status, {"input_tokens": 9, "output_tokens": 2}, error={"message": "failed"} if status == "failed" else None)
                wire = sse({"type": "response." + status, "response": result})
                with upstream(response_writer(wire, content_type="text/event-stream")) as (url, calls):
                    proxy = ResponsesProxy(url, ModelSettings("model"))
                    with proxy:
                        self.assertEqual(request(proxy)[2], wire)
                    receipt = proxy.receipts[0]
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(receipt.success)
                    self.assertEqual(receipt.terminal_status, status)
                    self.assertEqual((receipt.input_tokens, receipt.output_tokens), (9, 2))
                    self.assertIsNone(receipt.total_tokens)

    def test_http_error_bytes_and_usage_are_preserved_without_retry(self):
        wire = b'{"error":{"message":"rate limited"},"usage":{"input_tokens":7}}'
        with upstream(response_writer(wire, status=429, headers={"Retry-After": "60"})) as (url, calls):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            with proxy:
                status, headers, received = request(proxy)
            self.assertEqual((status, received), (429, wire))
            self.assertEqual(headers["Retry-After"], "60")
            self.assertEqual(len(calls), 1)
            self.assertEqual(proxy.receipts[0].input_tokens, 7)
            self.assertFalse(proxy.receipts[0].success)

    def test_redirect_is_returned_and_never_followed(self):
        wire = b"Moved by provider"
        with upstream(response_writer(wire, status=307, headers={"Location": "/v1/responses"})) as (url, calls):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            with proxy:
                status, headers, received = request(proxy)
            self.assertEqual((status, received), (307, wire))
            self.assertNotIn("Location", headers)
            self.assertEqual(len(calls), 1)
            self.assertEqual(proxy.receipts[0].http_status, 307)
            self.assertEqual(proxy.receipts[0].error, "HTTPError")

    def test_preflight_rejections_never_count_as_provider_requests(self):
        with upstream() as (url, calls):
            proxy = ResponsesProxy(url, ModelSettings("model"), allowed_tools=set())
            with proxy:
                self.assertEqual(request(proxy, {"model": "other"})[0], 400)
                self.assertEqual(request(proxy, {"model": "other"}, "/v1/responses/compact")[0], 400)
                self.assertEqual(request(proxy, path="/v1/chat/completions")[0], 404)
                self.assertEqual(request(proxy, path="/v1/responses?forward=elsewhere")[0], 404)
                self.assertEqual(request(proxy, headers={"Authorization": "Bearer desktop-key"})[0], 401)
                for tool in ({"type": "function", "name": "shell"}, {"type": "web_search"},
                             {"type": "namespace", "name": "evobridge", "tools": []}):
                    self.assertEqual(request(proxy, {"tools": [tool]})[0], 400)
                self.assertEqual(request(proxy, b'{"model":"model","model":"other"}')[0], 400)
                self.assertEqual(request(proxy, b'{"input":NaN}')[0], 400)
                self.assertEqual(request(proxy, headers={"Content-Encoding": "gzip"})[0], 400)
            self.assertEqual(calls, [])
            self.assertEqual(proxy.receipts, [])

    def test_missing_env_credential_is_preflight_rejection(self):
        with upstream() as (url, calls):
            with patch.dict(os.environ, {"EVOBRIDGE_ABSENT_KEY": ""}):
                proxy = ResponsesProxy(url, ModelSettings("model"), "EVOBRIDGE_ABSENT_KEY")
                with proxy:
                    self.assertEqual(request(proxy)[0], 400)
            self.assertEqual(calls, [])
            self.assertEqual(proxy.receipts, [])

    def test_connection_failure_is_one_actual_attempt(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proxy = ResponsesProxy(f"http://127.0.0.1:{port}/v1", ModelSettings("model"))
        with proxy:
            self.assertEqual(request(proxy)[0], 502)
        self.assertEqual(len(proxy.receipts), 1)
        self.assertTrue(proxy.receipts[0].attempted)
        self.assertFalse(proxy.receipts[0].success)
        self.assertIsNone(proxy.receipts[0].input_tokens)

    def test_provider_header_and_stream_waits_have_no_generation_deadline(self):
        wire = sse({"type": "response.completed", "response": terminal()})
        writer = response_writer(wire, content_type="text/event-stream", header_delay=.08, body_delay=.08)
        with upstream(writer) as (url, calls):
            proxy = ResponsesProxy(url, ModelSettings("model"), connection_timeout=.02)
            with proxy:
                self.assertEqual(request(proxy)[2], wire)
            self.assertTrue(proxy.receipts[0].success)
            self.assertGreater(proxy.receipts[0].elapsed_seconds, .12)
            self.assertEqual(len(calls), 1)

    def test_stream_is_forwarded_before_terminal_and_drained_after_client_disconnect(self):
        first = sse({"type": "response.output_text.delta", "delta": "first chunk"})
        last = sse({"type": "response.completed", "response": terminal(usage={"input_tokens": 3})})
        release = threading.Event()

        def streaming(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(first)
            handler.wfile.flush()
            release.wait(2)
            handler.wfile.write(last)
            handler.wfile.flush()

        with upstream(streaming) as (url, _):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            with proxy:
                conn, response = open_response(proxy)
                try:
                    self.assertEqual(response.read1(len(first)), first)
                    self.assertEqual(len(proxy.receipts), 1)
                    self.assertTrue(proxy.receipts[0].attempted)
                finally:
                    response.close()
                    conn.close()
                    release.set()
            self.assertEqual(proxy.receipts[0].input_tokens, 3)
            self.assertTrue(proxy.receipts[0].success)

    def test_exceptional_exit_interrupts_owned_sockets_and_joins_handlers(self):
        first = sse({"type": "response.output_text.delta", "delta": "still generating"})
        release = threading.Event()

        def streaming(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(first)
            handler.wfile.flush()
            release.wait(3)

        with upstream(streaming) as (url, calls):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            conn = response = None
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "native died"):
                    with proxy:
                        conn, response = open_response(proxy)
                        self.assertEqual(response.read1(len(first)), first)
                        server = proxy._server
                        raise RuntimeError("native died")
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(proxy._connections, {})
                self.assertEqual(server._workers, {})
                self.assertEqual(len(calls), 1)
                self.assertFalse(proxy.receipts[0].success)
                self.assertEqual(proxy.receipts[0].error, "Cancelled")
                self.assertIsNone(proxy.receipts[0].total_tokens)
            finally:
                release.set()
                if response:
                    response.close()
                if conn:
                    conn.close()

    def test_missing_usage_is_not_zero_or_estimated(self):
        with upstream(response_writer(terminal()), response_writer(terminal(usage={
                "input_tokens": True, "output_tokens": -1, "total_tokens": "4"}))) as (url, _):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            with proxy:
                request(proxy)
                request(proxy)
            self.assertEqual(len(proxy.receipts), 2)
            for receipt in proxy.receipts:
                self.assertTrue(receipt.success)
                for name in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
                    self.assertIsNone(getattr(receipt, name))

    def test_stream_without_terminal_is_not_reported_successful(self):
        wire = sse({"type": "response.output_text.delta", "delta": "unfinished"})
        with upstream(response_writer(wire, content_type="text/event-stream")) as (url, _):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            with proxy:
                self.assertEqual(request(proxy)[2], wire)
            self.assertFalse(proxy.receipts[0].success)
            self.assertEqual(proxy.receipts[0].error, "MissingTerminal")

    def test_compressed_response_bytes_are_never_rewritten(self):
        wire = gzip.compress(json.dumps(terminal()).encode())
        with upstream(response_writer(wire, headers={"Content-Encoding": "gzip"})) as (url, _):
            proxy = ResponsesProxy(url, ModelSettings("model"))
            with proxy:
                status, headers, received = request(proxy)
            self.assertEqual((status, received), (200, wire))
            self.assertEqual(headers["Content-Encoding"], "gzip")
            self.assertIsNone(proxy.receipts[0].input_tokens)
            self.assertEqual(proxy.receipts[0].error, "UnsupportedAccountingEncoding")

    def test_context_lifecycle_and_endpoint_validation(self):
        for url in ("ftp://host/v1", "http://name:secret@host/v1", "http://host/v1?key=secret",
                    "http://host/v1#fragment", "http://host:99999", "http://host/v1/chat/completions"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ResponsesProxy(url, ModelSettings("model"))
        proxy = ResponsesProxy("http://localhost/v1", ModelSettings("model"))
        with self.assertRaises(RuntimeError):
            _ = proxy.base_url
        with proxy:
            self.assertTrue(proxy.base_url.startswith("http://127.0.0.1:"))
            self.assertGreater(len(proxy.token), 30)
            with self.assertRaises(RuntimeError):
                proxy.__enter__()
        with self.assertRaises(RuntimeError):
            _ = proxy.base_url


class IncrementalAccountingTests(unittest.TestCase):
    def test_large_single_terminal_line_keeps_only_bounded_accounting(self):
        observer = _Accounting("text/event-stream")
        observer.feed(b'data: {"type":"response.completed","response":{"output":[{"text":"')
        # Output looks like accounting JSON but is inside a generated string.
        fragment = b'not usage: \\"input_tokens\\": 9000, ' * 300
        for _ in range(300):
            observer.feed(fragment)
            self.assertLessEqual(len(observer.parser.token), 256)
            self.assertLessEqual(len(observer.prefix), 6)
        observer.feed(b'"}],"status":"completed","usage":{"input_tokens":13,"output_tokens":17}}}\n\n')
        fields = observer.finish()
        self.assertEqual((fields["input_tokens"], fields["output_tokens"]), (13, 17))
        self.assertEqual(fields["status"], "completed")

    def test_sse_split_at_each_byte_and_multiline_data(self):
        wire = (b"event: response.incomplete\r\n" +
                'data: {"response":{"output":[{"text":"你好"}],\r\n'.encode() +
                b'data: "usage":{"input_tokens":5,"output_tokens":2}}}\r\n\r\n')
        observer = _Accounting("text/event-stream; charset=utf-8")
        for byte in wire:
            observer.feed(bytes([byte]))
        fields = observer.finish()
        self.assertEqual(fields["status"], "incomplete")
        self.assertEqual(fields["input_tokens"], 5)

    def test_json_scalar_tokens_and_nested_output_do_not_spoof_usage(self):
        value = terminal(usage={"input_tokens": 21, "output_tokens": 1})
        value["output"] = [{"usage": {"input_tokens": 999}, "text": '\\"usage\\": {"input_tokens": 999}'}]
        observer = _Accounting("application/json")
        for byte in json.dumps(value).encode():
            observer.feed(bytes([byte]))
        fields = observer.finish()
        self.assertEqual(fields["input_tokens"], 21)
        self.assertEqual(fields["output_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
