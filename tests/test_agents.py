"""Model/HTTP contracts exercised with deterministic, injected transports."""
from dataclasses import replace
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from evobridge.agents import ModelCloudImprover, ModelLocalImprover, ModelWorker
from evobridge.checks import CheckSuite, RegisteredCheck
from evobridge.config import ModelSettings
from evobridge.harness import Candidate, Harness
from evobridge.providers import HTTPChatClient, TokenBudget, conservative_token_count
from evobridge.types import CheckResult, Experience, LocalContext, Message, ProtocolError, Task


BASE = Harness({"memory.md": "Known facts", "CONTEXT.md": "Follow relevant skills.",
                "skills/old/SKILL.md": "An obsolete procedure"})
CONTEXT = LocalContext(BASE, Experience("e1", "private task", "private result"),
                       (Experience("past", "earlier private task", "past result"),),
                       {"manual": "private resource contents"})


def tool(tool_name, **arguments):
    return {"action": "tool", "name": tool_name, "arguments": arguments}


def final(content):
    return {"action": "final", "content": content}


class FakeClient:
    def __init__(self, *responses, settings=None):
        self.settings = settings or ModelSettings.local()
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        reply = self.responses.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


class ProviderTests(unittest.TestCase):
    def test_explicit_sampling_and_environment_credential_are_sent(self):
        calls = []

        def transport(request, timeout):
            calls.append((request, timeout))
            return {"id": "request-1", "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}

        client = HTTPChatClient("http://localhost:8000/v1", ModelSettings.local(),
                                "TEST_EVOBRIDGE_KEY", 15, transport=transport)
        with patch.dict(os.environ, {"TEST_EVOBRIDGE_KEY": "test-only-key"}):
            self.assertEqual(client.complete([Message("user", "hello")]), "answer")
        request, timeout = calls[0]
        self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-key")
        self.assertEqual(timeout, 15)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "Qwen3.5-9B")
        self.assertEqual(body["temperature"], .6)
        self.assertEqual(body["top_p"], .95)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["min_p"], 0)
        self.assertEqual(body["presence_penalty"], 0)
        self.assertEqual(body["repetition_penalty"], 1)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["max_completion_tokens"], 32768)
        receipt = client.usage_receipts[0]
        self.assertTrue(receipt.success)
        self.assertTrue(receipt.attempted)
        self.assertGreaterEqual(receipt.elapsed_seconds, 0)
        self.assertEqual((receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens), (12, 3, 15))
        self.assertEqual(receipt.request_id, "request-1")

    def test_cloud_effort_and_missing_usage_are_explicit(self):
        bodies = []

        def transport(request, timeout):
            bodies.append(json.loads(request.data))
            return json.dumps({"choices": [{"message": {"content": "cloud answer"}}]}).encode()

        client = HTTPChatClient("https://example.invalid/v1/chat/completions",
                                ModelSettings("cloud", reasoning_effort="medium"), transport=transport)
        client.complete([{"role": "user", "content": "guarded summary"}])
        self.assertEqual(bodies[0]["reasoning_effort"], "medium")
        self.assertNotIn("chat_template_kwargs", bodies[0])
        self.assertIsNone(client.usage_receipts[0].prompt_tokens)
        self.assertIsNone(client.usage_receipts[0].total_tokens)

    def test_response_error_preserves_known_usage_and_never_retries(self):
        calls = []

        def transport(request, timeout):
            calls.append(request)
            return {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 20}}

        client = HTTPChatClient("https://example.invalid/v1", ModelSettings("model"), transport=transport)
        with self.assertRaises(ProtocolError):
            client.complete([Message("user", "request")])
        self.assertEqual(len(calls), 1)
        receipt = client.usage_receipts[0]
        self.assertFalse(receipt.success)
        self.assertEqual(receipt.prompt_tokens, 40)
        self.assertEqual(receipt.completion_tokens, 20)
        self.assertIsNone(receipt.total_tokens)

    def test_http_failure_retains_receipt_without_replaying(self):
        calls = []

        def transport(request, timeout):
            calls.append(request)
            raise HTTPError(request.full_url, 429, "throttled", {},
                            io.BytesIO(b'{"usage":{"prompt_tokens":7},"error":"throttled"}'))

        client = HTTPChatClient("https://example.invalid/v1", ModelSettings("model"), transport=transport)
        with self.assertRaises(HTTPError):
            client.complete([Message("user", "request")])
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.usage_receipts[0].prompt_tokens, 7)
        self.assertEqual(client.usage_receipts[0].error, "HTTPError")

    def test_oversized_mandatory_input_fails_before_transport(self):
        calls = []
        settings = ModelSettings("model", context_tokens=500, compact_at=400, max_output_tokens=100)
        client = HTTPChatClient("http://localhost/v1", settings,
                                transport=lambda *args: calls.append(args))
        with self.assertRaises(ProtocolError):
            client.complete([Message("system", "S" * 600), Message("user", "latest")])
        self.assertEqual(calls, [])
        self.assertFalse(client.usage_receipts[0].success)
        self.assertFalse(client.usage_receipts[0].attempted)

    def test_missing_credential_is_a_local_rejection_not_an_http_attempt(self):
        calls = []
        client = HTTPChatClient("https://example.invalid", ModelSettings("model"),
                                "TEST_EVOBRIDGE_ABSENT", transport=lambda *args: calls.append(args))
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            client.complete([Message("user", "request")])
        self.assertEqual(calls, [])
        self.assertFalse(client.usage_receipts[0].attempted)
        self.assertIsNone(client.usage_receipts[0].prompt_tokens)

    def test_cached_and_reasoning_usage_are_subsets_not_extra_total_tokens(self):
        def transport(request, timeout):
            return {"choices": [{"message": {"content": "answer"}}], "usage": {
                "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 30}}}

        client = HTTPChatClient("https://example.invalid", ModelSettings("model"), transport=transport)
        client.complete([Message("user", "request")])
        receipt = client.usage_receipts[0]
        self.assertEqual(receipt.prompt_cached_tokens, 80)
        self.assertEqual(receipt.completion_reasoning_tokens, 30)
        self.assertEqual(receipt.total_tokens, 140)

    def test_transport_failure_receipt_does_not_invent_token_counts(self):
        def transport(request, timeout):
            raise TimeoutError("private provider text")

        client = HTTPChatClient("https://example.invalid", ModelSettings("model"), transport=transport)
        with self.assertRaises(TimeoutError):
            client.complete([Message("user", "request")])
        receipt = client.usage_receipts[0]
        self.assertTrue(receipt.attempted)
        self.assertEqual(receipt.error, "TimeoutError")
        self.assertIsNone(receipt.prompt_tokens)
        self.assertIsNone(receipt.completion_tokens)


class BudgetTests(unittest.TestCase):
    def test_compaction_preserves_both_system_protocols_and_latest_input(self):
        settings = ModelSettings("model", context_tokens=1200, compact_at=800, max_output_tokens=200)
        budget = TokenBudget(settings)
        original = [{"role": "system", "content": "Trust boundary"},
                    {"role": "system", "content": "Current phase protocol"},
                    {"role": "user", "content": "Protected task and phase payload"},
                    {"role": "user", "content": "old history " * 100},
                    {"role": "assistant", "content": "old tool request"},
                    {"role": "user", "content": "latest tool response"}]
        fitted, omitted = budget.fit(original)
        self.assertGreater(omitted, 0)
        self.assertEqual(fitted[:2], original[:2])
        self.assertEqual(fitted[2], original[2])
        self.assertEqual(fitted[-1], original[-1])
        self.assertIn("omitted", json.dumps(fitted))
        self.assertLessEqual(budget.count_messages(fitted), 800)
        self.assertEqual(len(original), 6)
        refitted, _ = budget.fit(fitted + [{"role": "assistant", "content": "old " * 250},
                                          {"role": "user", "content": "newest evidence"}])
        self.assertEqual(refitted[2], original[2])
        self.assertEqual(refitted[-1]["content"], "newest evidence")

    def test_utf8_tool_truncation_keeps_valid_json_and_announces_loss(self):
        settings = replace(ModelSettings.local(), tool_output_tokens=256)
        budget = TokenBudget(settings)
        encoded = budget.tool_result("read", "秘密\\\"\n" * 1000)
        decoded = json.loads(encoded)
        self.assertTrue(decoded["truncated"])
        self.assertLessEqual(conservative_token_count(encoded), 256)
        self.assertNotIn("result", decoded)

    def test_injected_counter_is_used_and_invalid_counts_fail(self):
        seen = []

        def counter(text):
            seen.append(text)
            return len(text)

        budget = TokenBudget(ModelSettings.local(), counter)
        self.assertEqual(budget.count("中文"), 2)
        self.assertEqual(seen, ["中文"])
        self.assertEqual(budget.counter_name, "injected_tokenizer")
        with self.assertRaises(ValueError):
            TokenBudget(ModelSettings.local(), lambda text: True).count("text")


class LocalAgentTests(unittest.TestCase):
    def test_private_tools_support_preparation_and_persist_into_inquiry(self):
        client = FakeClient(tool("read", source="history", path="past"),
                            tool("read", source="resource", path="manual"),
                            final("focused public summary"),
                            tool("read", source="harness", path="memory.md"), final("targeted answer"))
        local = ModelLocalImprover(client)
        self.assertEqual(local.prepare(CONTEXT), "focused public summary")
        self.assertEqual(local.investigate("Which fact is missing?", CONTEXT), "targeted answer")
        last = json.dumps(client.calls[-1])
        for expected in ("earlier private task", "private resource contents", "focused public summary", "Known facts"):
            self.assertIn(expected, last)

    def test_actual_candidate_edits_and_registered_check_are_observed(self):
        check_calls = []

        def check(candidate):
            check_calls.append(candidate)
            return CheckResult("facts", candidate.read("memory.md") == "New fact", "checked contents")

        client = FakeClient(tool("write", path="memory.md", content="New fact"),
                            tool("delete", path="skills/old/SKILL.md"),
                            tool("check", name="facts"),
                            tool("read", source="candidate", path="memory.md"),
                            {"action": "final", "status": "READY", "reviewed": True,
                             "review": "Checked the changed fact and removed the obsolete operation."})
        local = ModelLocalImprover(client, CheckSuite({"facts": RegisteredCheck(check)}))
        candidate = Candidate(BASE)
        result = local.realize(candidate, (Message("assistant", "Update the fact."),), CONTEXT, 3)
        self.assertEqual(result.status, "READY")
        self.assertIs(result.reviewed, True)
        self.assertIsNot(check_calls[0], candidate)
        self.assertEqual(check_calls[0].freeze().id, candidate.freeze().id)
        self.assertEqual(check_calls[0].changed_paths, candidate.changed_paths)
        self.assertEqual(candidate.read("memory.md"), "New fact")
        self.assertNotIn("skills/old/SKILL.md", candidate.list_files())
        self.assertEqual(BASE.files["memory.md"], "Known facts")
        self.assertIn("checked contents", json.dumps(client.calls[-1]))

    def test_readonly_inquiry_rejects_writes_even_if_candidate_is_available(self):
        for method in ("prepare", "investigate"):
            with self.subTest(method=method):
                client = FakeClient(tool("write", path="memory.md", content="unauthorized"))
                local = ModelLocalImprover(client)
                candidate = Candidate(BASE)
                with self.assertRaises(ProtocolError):
                    if method == "prepare":
                        local.prepare(CONTEXT)
                    else:
                        local.investigate("question", CONTEXT, candidate)
                self.assertEqual(candidate.freeze().id, BASE.id)
                self.assertEqual(len(client.calls), 1)

    def test_unregistered_shell_or_check_fails_closed(self):
        for request in (tool("shell", command="echo bad"), tool("check", name="unregistered")):
            client = FakeClient(request)
            with self.assertRaises(ProtocolError):
                ModelLocalImprover(client).realize(Candidate(BASE), (Message("user", "proposal"),), CONTEXT, 1)
            self.assertEqual(len(client.calls), 1)

    def test_inquiry_checks_inspect_detached_copy_without_mutating_active_candidate(self):
        checked = []

        def bad_check(candidate):
            checked.append(candidate)
            candidate.write("memory.md", "accidental check mutation")
            return CheckResult("inspect", True, "looked at candidate")

        client = FakeClient(tool("check", name="inspect"), final("The check rejected its own mutation."))
        local = ModelLocalImprover(client, CheckSuite({"inspect": RegisteredCheck(bad_check)}))
        active = Candidate(BASE)
        active.write("memory.md", "active work in progress")
        local.investigate("Inspect the current draft", CONTEXT, active)
        self.assertIsNot(checked[0], active)
        self.assertEqual(checked[0].changed_paths, active.changed_paths)
        self.assertEqual(active.read("memory.md"), "active work in progress")
        self.assertEqual(BASE.files["memory.md"], "Known facts")
        result = json.loads(client.calls[-1][-1]["content"])["result"]
        self.assertFalse(result["passed"])

    def test_inquiry_checks_without_candidate_use_a_copy_of_published_harness(self):
        seen = []

        def inspect(candidate):
            seen.append(candidate.read("memory.md"))
            return CheckResult("inspect", True, "inspected")

        client = FakeClient(tool("check", name="inspect"), final("Inspected published facts."))
        ModelLocalImprover(client, CheckSuite({"inspect": RegisteredCheck(inspect)})).investigate(
            "Inspect published behavior", CONTEXT)
        self.assertEqual(seen, ["Known facts"])

    def test_malformed_protocol_is_not_repaired_or_coerced(self):
        responses = ['```json\n{"action":"final","content":"text"}\n```',
                     '{"action":"final","action":"tool","content":"text"}',
                     {"action": "final", "content": "text", "extra": 1},
                     '["final"]', '{"action":"final","content":NaN}']
        for reply in responses:
            with self.subTest(reply=reply):
                client = FakeClient(reply)
                with self.assertRaises(ProtocolError):
                    ModelLocalImprover(client).prepare(CONTEXT)
                self.assertEqual(len(client.calls), 1)

    def test_clarification_continues_same_candidate_and_private_history(self):
        client = FakeClient(tool("write", path="memory.md", content="intermediate"),
                            {"action": "final", "status": "CLARIFY", "clarification": "Use the fallback?"},
                            tool("read", source="candidate", path="memory.md"),
                            {"action": "final", "status": "STOP", "review": "Proposal withdrawn."})
        local = ModelLocalImprover(client)
        candidate = Candidate(BASE)
        first = local.realize(candidate, (Message("user", "proposal"),), CONTEXT, 3)
        second = local.realize(candidate, (Message("user", "proposal"), Message("assistant", "withdraw")), CONTEXT, 2)
        self.assertEqual(first.status, "CLARIFY")
        self.assertEqual(second.status, "STOP")
        self.assertEqual(candidate.read("memory.md"), "intermediate")
        self.assertIn("Use the fallback?", json.dumps(client.calls[-1]))

    def test_model_cannot_claim_review_with_a_truthy_string(self):
        client = FakeClient({"action": "final", "status": "READY", "reviewed": "true", "review": "review"})
        with self.assertRaises(ProtocolError):
            ModelLocalImprover(client).realize(Candidate(BASE), (Message("user", "proposal"),), CONTEXT, 1)

    def test_tool_loop_has_a_finite_budget(self):
        client = FakeClient(tool("list", source="harness"), tool("list", source="harness"))
        with self.assertRaises(ProtocolError):
            ModelLocalImprover(client, max_tool_steps=1).prepare(CONTEXT)
        self.assertEqual(len(client.calls), 2)

    def test_long_experience_uses_bounded_overview_and_paging_can_read_its_tail(self):
        experience = Experience("long-record", "Preserved task description", "outcome",
                                ("prefix" * 40000, "LATE_EXPERIENCE_SENTINEL"), "feedback")
        serialized = json.dumps(experience.to_dict(), ensure_ascii=False)
        offset = serialized.index("LATE_EXPERIENCE_SENTINEL")
        context = LocalContext(BASE, experience, (), {})
        settings = replace(ModelSettings.local(), context_tokens=16000, compact_at=12000,
                           max_output_tokens=2000, tool_output_tokens=512)
        client = FakeClient(tool("read", source="experience", offset=offset, limit=100),
                            final("Inspected the late observation."), settings=settings)
        local = ModelLocalImprover(client)
        local.prepare(context)
        self.assertIn("Preserved task description", client.calls[0][2]["content"])
        self.assertNotIn("prefix" * 100, json.dumps(client.calls[0]))
        self.assertIn("full_record_access", json.dumps(client.calls[0]))
        page = json.loads(client.calls[-1][-1]["content"])["result"]
        self.assertIn("LATE_EXPERIENCE_SENTINEL", page["content"])
        self.assertEqual(page["offset"], offset)
        self.assertEqual(page["total_chars"], len(serialized))
        self.assertEqual(page["format"], "json")

    def test_all_local_read_sources_can_reach_content_beyond_the_initial_preview(self):
        value = "prefix" * 10000 + "LATE_CONTENT_SENTINEL"
        harness = Harness({**BASE.files, "skills/large/SKILL.md": value})
        historical = Experience("old-record", "task", value)
        history_json = json.dumps(historical.to_dict(), ensure_ascii=False)
        context = LocalContext(harness, CONTEXT.experience, (historical,), {"large": value})
        sources = [("harness", "skills/large/SKILL.md", value),
                   ("candidate", "skills/large/SKILL.md", value),
                   ("resource", "large", value), ("history", "old-record", history_json)]
        for source, path, complete_text in sources:
            with self.subTest(source=source):
                offset = complete_text.index("LATE_CONTENT_SENTINEL")
                client = FakeClient(tool("read", source=source, path=path, offset=offset, limit=100),
                                    final("Read late evidence."),
                                    settings=replace(ModelSettings.local(), tool_output_tokens=256))
                ModelLocalImprover(client).investigate("Read the tail", context, Candidate(harness))
                encoded = client.calls[-1][-1]["content"]
                page = json.loads(encoded)["result"]
                self.assertIn("LATE_CONTENT_SENTINEL", page["content"])
                self.assertLessEqual(conservative_token_count(encoded), 256)
                self.assertEqual(page["total_chars"], len(complete_text))
                self.assertNotIn("truncated", json.loads(encoded))

    def test_pages_shrink_to_budget_and_next_offsets_reconstruct_the_exact_source(self):
        value = ('前缀\\"\n' * 200) + "LATE_SENTINEL"
        context = LocalContext(BASE, CONTEXT.experience, (), {"paged": value})

        class PagingClient(FakeClient):
            def __init__(self):
                super().__init__(settings=replace(ModelSettings.local(), tool_output_tokens=384))
                self.segments = []

            def complete(self, messages):
                self.calls.append([dict(m) for m in messages])
                latest = json.loads(messages[-1]["content"])
                offset = 0
                if latest.get("tool") == "read":
                    self.segments.append(latest["result"]["content"])
                    offset = latest["result"]["next_offset"]
                    if offset is None:
                        return json.dumps(final("Read the entire private resource."))
                return json.dumps(tool("read", source="resource", path="paged", offset=offset, limit=65536))

        client = PagingClient()
        ModelLocalImprover(client).prepare(context)
        self.assertEqual("".join(client.segments), value)
        self.assertGreater(len(client.segments), 1)
        for call in client.calls[1:]:
            envelope = call[-1]["content"]
            self.assertLessEqual(conservative_token_count(envelope), 384)
            self.assertNotIn("truncated", json.loads(envelope))

    def test_invalid_page_parameters_fail_closed_without_additional_model_calls(self):
        for paging in ({"offset": -1}, {"offset": True}, {"offset": "1"},
                       {"limit": 0}, {"limit": 65537}, {"limit": False},
                       {"limit": 1.5}, {"offset": 10000, "limit": 10}):
            with self.subTest(paging=paging):
                client = FakeClient(tool("read", source="harness", path="memory.md", **paging))
                with self.assertRaises(ProtocolError):
                    ModelLocalImprover(client).prepare(CONTEXT)
                self.assertEqual(len(client.calls), 1)

    def test_too_small_budget_rejects_page_instead_of_losing_continuation_metadata(self):
        client = FakeClient(tool("read", source="harness", path="memory.md", offset=0, limit=10),
                            settings=replace(ModelSettings.local(), tool_output_tokens=32))
        with self.assertRaises(ProtocolError):
            ModelLocalImprover(client).prepare(CONTEXT)
        self.assertEqual(len(client.calls), 1)


class CloudAndWorkerTests(unittest.TestCase):
    def test_identical_briefs_do_not_reuse_hidden_cloud_answers_between_rounds(self):
        client = FakeClient({"action": "PROPOSE", "content": "old round unique proposal"},
                            {"action": "RETAIN", "content": "fresh round decision"})
        cloud = ModelCloudImprover(client)
        same_brief = (Message("user", "identical summary"),)
        cloud.decide(same_brief, 3)
        cloud.decide(same_brief, 3)
        self.assertNotIn("old round unique proposal", json.dumps(client.calls[-1]))
        self.assertEqual(client.calls[0], client.calls[1])

    def test_cloud_uses_authoritative_transcript_once_without_duplicating_own_answer(self):
        client = FakeClient({"action": "ASK", "content": "unique question"},
                            {"action": "PROPOSE", "content": "proposal"})
        cloud = ModelCloudImprover(client)
        initial = (Message("user", "summary"),)
        cloud.decide(initial, 3)
        supplied = initial + (Message("assistant", "unique question"), Message("user", "answer"))
        cloud.decide(supplied, 2)
        prompt = json.dumps(client.calls[-1])
        self.assertEqual(prompt.count("unique question"), 1)
        self.assertEqual(json.loads(client.calls[-1][-1]["content"])["transcript"],
                         [{"role": m.role, "content": m.content} for m in supplied])

    def test_cloud_has_only_guarded_input_and_keeps_prior_exchange(self):
        client = FakeClient({"action": "ASK", "content": "Which behavior failed?"},
                            {"action": "PROPOSE", "content": "Update the memory."},
                            {"action": "RESPOND", "content": "Use the documented fallback."})
        cloud = ModelCloudImprover(client)
        initial = (Message("user", "BLOCKED"),)
        self.assertEqual(cloud.decide(initial, 3).action, "ASK")
        supplied = initial + (Message("assistant", "Which behavior failed?"), Message("user", "sanitized evidence"))
        self.assertEqual(cloud.decide(supplied, 2).action, "PROPOSE")
        self.assertEqual(cloud.respond(supplied + (Message("user", "A clarification"),), 2).action, "RESPOND")
        prompt = json.dumps(client.calls[-1])
        self.assertIn("BLOCKED", prompt)
        self.assertIn("sanitized evidence", prompt)
        self.assertNotIn("private resource contents", prompt)
        self.assertNotIn("private task", prompt)

    def test_cloud_enforces_phase_actions_fields_and_inquiry_budget(self):
        cases = [("decide", {"action": "RESPOND", "content": "answer"}, 1),
                 ("respond", {"action": "RETAIN", "content": "retain"}, 1),
                 ("decide", {"action": "ASK", "content": "question"}, 0),
                 ("decide", {"action": "PROPOSE", "content": "proposal", "private": "data"}, 1)]
        for phase, reply, budget in cases:
            with self.subTest(phase=phase, reply=reply):
                client = FakeClient(reply)
                with self.assertRaises(ProtocolError):
                    getattr(ModelCloudImprover(client), phase)((Message("user", "summary"),), budget)
                self.assertEqual(len(client.calls), 1)

    def test_worker_initializes_context_and_uses_only_registered_host_tools(self):
        def lookup(key):
            """Look up a public fixture key using argument key."""
            return {"value": "fixture:" + key}

        client = FakeClient(tool("read_harness", path="memory.md"), tool("lookup", key="sample"), final("task answer"))
        worker = ModelWorker(client, {"lookup": lookup})
        result = worker.execute(Task("task1", "Answer the task"), BASE)
        self.assertEqual(result.outcome, "task answer")
        self.assertEqual(result.feedback, "")
        self.assertEqual(len(result.observations), 2)
        self.assertIn("fixture:sample", result.observations[-1])
        self.assertIn("Follow relevant skills.", client.calls[0][-1]["content"])
        self.assertEqual(BASE.files["memory.md"], "Known facts")

    def test_worker_cannot_request_an_implicit_environment_tool(self):
        client = FakeClient(tool("shell", command="echo unavailable"))
        with self.assertRaises(ProtocolError):
            ModelWorker(client).execute(Task("task1", "Run a task"), BASE)
        self.assertEqual(len(client.calls), 1)

    def test_worker_read_harness_can_page_to_a_late_sentinel(self):
        prefix = "large artifact prefix " * 10000
        harness = Harness({**BASE.files, "skills/large/SKILL.md": prefix + "LATE_WORKER_SENTINEL"})
        client = FakeClient(tool("read_harness", path="skills/large/SKILL.md", offset=len(prefix), limit=100),
                            final("Found the late artifact content."),
                            settings=replace(ModelSettings.local(), tool_output_tokens=256))
        experience = ModelWorker(client).execute(Task("t", "Inspect the artifact tail"), harness)
        page = json.loads(experience.observations[0])["result"]
        self.assertEqual(page["content"], "LATE_WORKER_SENTINEL")
        self.assertIsNone(page["next_offset"])
        self.assertLessEqual(conservative_token_count(experience.observations[0]), 256)

    def test_worker_compaction_retains_task_context_and_latest_tool_result(self):
        settings = ModelSettings("model", context_tokens=4500, compact_at=3500,
                                 max_output_tokens=500, tool_output_tokens=2200)
        client = FakeClient(tool("large"), tool("large"), final("answer"), settings=settings)
        worker = ModelWorker(client, {"large": lambda: "evidence" * 200})
        worker.execute(Task("task", "protected original task"), BASE)
        final_prompt = client.calls[-1]
        self.assertIn("Context compaction", json.dumps(final_prompt))
        self.assertIn("protected original task", final_prompt[2]["content"])
        self.assertIn("Follow relevant skills.", final_prompt[2]["content"])
        self.assertIn("evidence", final_prompt[-1]["content"])


if __name__ == "__main__":
    unittest.main()
