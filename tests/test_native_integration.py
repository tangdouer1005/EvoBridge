"""Optional installed-runtime integration against a local synthetic model server.

Run EVOBRIDGE_TEST_NATIVE=1 python -m unittest discover -s tests -p test_native_integration.py -v.
No external model endpoint, credential, or benchmark is used.
"""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from evobridge.config import ModelSettings
from evobridge.runtime import CodexRuntime, CodexRuntimeError


class SyntheticResponses(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        n = len(self.server.requests)
        redirect = getattr(self.server, "redirect", None)
        if redirect:
            payload = b'{"error":{"message":"Synthetic redirect"}}'
            self.send_response(307)
            self.send_header("Location", redirect)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        response_id = f"resp_synthetic_{n}"
        if n == 1 and self.server.with_tool:
            item = {"type": "function_call", "id": "fc_synthetic", "call_id": "call_synthetic",
                    "name": "write", "arguments": json.dumps({"path": "memory.md", "content": "Checked fact"}),
                    "status": "completed"}
        else:
            item = {"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                    "phase": "final_answer", "content": [{"type": "output_text",
                    "text": '{"status":"READY","reviewed":true,"review":"Synthetic artifact review"}',
                    "annotations": []}]}
        response = {"id": response_id, "object": "response", "created_at": 0,
                    "model": body["model"], "status": "completed", "output": [item],
                    "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}
        events = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                  {"type": "response.output_item.added", "output_index": 0, "item": item},
                  {"type": "response.output_item.done", "output_index": 0, "item": item},
                  {"type": "response.completed", "response": response}]
        payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()


@unittest.skipUnless(os.getenv("EVOBRIDGE_TEST_NATIVE") == "1" and shutil.which("codex"),
                     "Optional installed agent runtime integration")
class InstalledRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticResponses)
        self.server.requests = []
        self.server.with_tool = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_native_loop_dispatches_tool_and_collects_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            runtime = CodexRuntime(f"http://127.0.0.1:{self.server.server_port}/v1",
                                   ModelSettings.local(), logdir=root, startup_timeout=45)
            calls = []
            def dispatch(name, args):
                calls.append((name, args))
                return {"written": args["path"]}
            answer = runtime.run("Use the granted write tool, then return the final JSON.", "Synthetic input.", [{
                "name": "write", "description": "Write a candidate text artifact.",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"},
                    "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}}], dispatch)
            self.assertEqual(json.loads(answer)["status"], "READY")
            self.assertEqual(calls, [("write", {"path": "memory.md", "content": "Checked fact"})])
            self.assertEqual(len(self.server.requests), 2)
            self.assertEqual(len(runtime.usage_receipts), 2)
            self.assertEqual(runtime.session_receipts[0].status, "completed")
            for request in self.server.requests:
                self.assertEqual(request["model"], ModelSettings.local().model)
                self.assertEqual(request["reasoning"]["effort"], "none")
                self.assertEqual(request["max_output_tokens"], 32768)
                self.assertEqual({tool["name"] for tool in request.get("tools", [])}, {"write"})
            self.assertIn("function_call_output", json.dumps(self.server.requests[1]))

    def test_cloud_session_has_no_tools_and_explicit_reasoning(self):
        self.server.with_tool = False
        runtime = CodexRuntime(f"http://127.0.0.1:{self.server.server_port}/v1",
                               ModelSettings("gpt6-sol", reasoning_effort="medium"), startup_timeout=45)
        runtime.run("Return final JSON only.", "A guarded transcript.", [],
                    lambda *_: self.fail("No tool should be dispatched"))
        self.assertEqual(len(self.server.requests), 1)
        request = self.server.requests[0]
        self.assertEqual(request.get("tools", []), [])
        self.assertEqual(request["reasoning"]["effort"], "medium")

    def test_native_client_cannot_follow_upstream_redirect(self):
        target = ThreadingHTTPServer(("127.0.0.1", 0), SyntheticResponses)
        target.requests = []
        target.with_tool = False
        thread = threading.Thread(target=target.serve_forever, daemon=True)
        thread.start()
        self.server.redirect = f"http://127.0.0.1:{target.server_port}/v1/responses"
        runtime = CodexRuntime(f"http://127.0.0.1:{self.server.server_port}/v1", ModelSettings.local())
        try:
            with self.assertRaises(CodexRuntimeError):
                runtime.run("Return a final JSON message.", "PRIVATE_REDIRECT_SENTINEL", [], lambda *_: None)
            self.assertEqual(len(self.server.requests), 1)
            self.assertEqual(target.requests, [])
            self.assertEqual(runtime.usage_receipts[0].http_status, 307)
        finally:
            target.shutdown()
            target.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
