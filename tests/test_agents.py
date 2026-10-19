"""Native role contracts, exercised without model calls or an installed server."""
from dataclasses import dataclass, replace
import json
import unittest

from evobridge.agents import HostTool, ModelCloudImprover, ModelLocalImprover, ModelWorker
from evobridge.checks import CheckSuite, RegisteredCheck
from evobridge.config import ModelSettings
from evobridge.harness import Candidate, Harness
from evobridge.providers import TokenBudget, conservative_token_count
from evobridge.types import CheckResult, Experience, LocalContext, Message, ProtocolError, Task


BASE = Harness({"memory.md": "Known facts", "CONTEXT.md": "Follow relevant skills.",
                "skills/old/SKILL.md": "An obsolete procedure"})
CONTEXT = LocalContext(BASE, Experience("e1", "private task", "private result"),
                       (Experience("past", "earlier private task", "past result"),),
                       {"manual": "private resource contents"})


def tool(tool_name, **arguments):
    """A test script calls the registered callback, not a model JSON action."""
    return tool_name, arguments


@dataclass
class NativePhase:
    final: object
    calls: tuple = ()


class FakeNativeRuntime:
    """One run per native phase; scripts invoke native dispatch callbacks directly."""

    def __init__(self, *phases, settings=None):
        self.settings = settings or ModelSettings.local()
        self.phases = list(phases)
        self.calls = []

    def run(self, instructions, prompt, tools, dispatch):
        call = {"instructions": instructions, "prompt": prompt,
                "tools": json.loads(json.dumps(tools)), "results": []}
        self.calls.append(call)

        def invoke(name, arguments):
            output = dispatch(name, arguments)
            call["results"].append({"name": name, "arguments": arguments, "output": output})
            return output

        phase = self.phases.pop(0)
        if callable(phase):
            final = phase(invoke)
        elif isinstance(phase, NativePhase):
            for name, arguments in phase.calls:
                invoke(name, arguments)
            final = phase.final
        else:
            final = phase
        return final if isinstance(final, str) else json.dumps(final)


def result(runtime, index=0, phase=-1):
    return json.loads(runtime.calls[phase]["results"][index]["output"])["result"]


def schemas(runtime, phase=-1):
    return {item["name"]: item["inputSchema"] for item in runtime.calls[phase]["tools"]}


class ToolBudgetTests(unittest.TestCase):
    def test_utf8_tool_truncation_keeps_valid_json_and_announces_loss(self):
        settings = replace(ModelSettings.local(), tool_output_tokens=256)
        encoded = TokenBudget(settings).tool_result("lookup", "秘密\\\"\n" * 1000)
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
        with self.assertRaises(ValueError):
            TokenBudget(ModelSettings.local(), lambda text: True).count("text")


class LocalAgentTests(unittest.TestCase):
    def test_native_tools_load_private_evidence_and_carry_only_explicit_phase_results(self):
        runtime = FakeNativeRuntime(
            NativePhase({"content": "focused public summary"}, (
                tool("read", source="history", path="past"), tool("read", source="resource", path="manual"))),
            NativePhase({"content": "targeted answer"}, (tool("read", source="harness", path="memory.md"),)))
        local = ModelLocalImprover(runtime)
        self.assertEqual(local.prepare(CONTEXT), "focused public summary")
        self.assertEqual(local.investigate("Which fact is missing?", CONTEXT), "targeted answer")
        self.assertEqual(len(runtime.calls), 2)
        self.assertIn("earlier private task", result(runtime, phase=0)["content"])
        self.assertEqual(result(runtime, index=1, phase=0)["content"], "private resource contents")
        self.assertIn("focused public summary", runtime.calls[1]["prompt"])
        self.assertNotIn("earlier private task", runtime.calls[1]["prompt"])
        self.assertNotIn("private resource contents", runtime.calls[1]["prompt"])
        self.assertEqual(result(runtime)["content"], "Known facts")
        self.assertEqual(set(schemas(runtime, 0)), {"read", "list"})
        self.assertNotIn('"action":"tool"', runtime.calls[0]["instructions"])

    def test_new_preparation_clears_previous_round_results(self):
        runtime = FakeNativeRuntime({"content": "old round summary unique"},
                                    {"content": "old inquiry answer unique"}, {"content": "new summary"})
        local = ModelLocalImprover(runtime)
        local.prepare(CONTEXT)
        local.investigate("old round question unique", CONTEXT)
        local.prepare(CONTEXT)
        prompt = json.loads(runtime.calls[-1]["prompt"])
        self.assertEqual(prompt["prior_phase_results"], [])
        self.assertNotIn("old round", runtime.calls[-1]["prompt"])

    def test_actual_candidate_edits_and_registered_check_are_observed(self):
        check_calls = []

        def check(candidate):
            check_calls.append(candidate)
            return CheckResult("facts", candidate.read("memory.md") == "New fact", "checked contents")

        runtime = FakeNativeRuntime(NativePhase(
            {"status": "READY", "reviewed": True, "review": "Checked changed facts and obsolete operations."}, (
                tool("write", path="memory.md", content="New fact"),
                tool("write", path="CONTEXT.md", content="Apply the updated fact."),
                tool("write", path="skills/new/SKILL.md", content="Use known facts."),
                tool("delete", path="skills/old/SKILL.md"), tool("check", name="facts"),
                tool("read", source="candidate", path="memory.md"))))
        local = ModelLocalImprover(runtime, CheckSuite({"facts": RegisteredCheck(check)}))
        candidate = Candidate(BASE)
        realized = local.realize(candidate, (Message("assistant", "Update the fact."),), CONTEXT, 3)
        self.assertEqual(realized.status, "READY")
        self.assertIs(realized.reviewed, True)
        self.assertIsNot(check_calls[0], candidate)
        self.assertEqual(check_calls[0].freeze().id, candidate.freeze().id)
        self.assertEqual(check_calls[0].changed_paths, candidate.changed_paths)
        self.assertEqual(candidate.read("memory.md"), "New fact")
        self.assertNotIn("skills/old/SKILL.md", candidate.list_files())
        self.assertEqual(BASE.files["memory.md"], "Known facts")
        self.assertEqual(result(runtime, 4)["details"], "checked contents")
        self.assertEqual(set(schemas(runtime)), {"read", "list", "check", "write", "delete"})
        self.assertEqual(schemas(runtime)["check"]["properties"]["name"]["enum"], ["facts"])

    def test_readonly_phase_neither_registers_nor_permits_mutation_tools(self):
        for method in ("prepare", "investigate"):
            with self.subTest(method=method):
                runtime = FakeNativeRuntime(NativePhase(None, (tool("write", path="memory.md", content="bad"),)))
                local = ModelLocalImprover(runtime)
                candidate = Candidate(BASE)
                with self.assertRaises(ProtocolError):
                    if method == "prepare":
                        local.prepare(CONTEXT)
                    else:
                        local.investigate("question", CONTEXT, candidate)
                self.assertEqual(candidate.freeze().id, BASE.id)
                self.assertNotIn("write", schemas(runtime))
                self.assertNotIn("delete", schemas(runtime))
                self.assertEqual(len(runtime.calls), 1)

    def test_editable_scope_excludes_host_files_and_protects_root_documents(self):
        for request in (tool("write", path="/tmp/private", content="x"),
                        tool("write", path="../escape", content="x"),
                        tool("write", path="README.md", content="x"),
                        tool("delete", path="CONTEXT.md")):
            with self.subTest(request=request):
                runtime = FakeNativeRuntime(NativePhase(None, (request,)))
                candidate = Candidate(BASE)
                with self.assertRaises(ValueError):
                    ModelLocalImprover(runtime).realize(candidate, (Message("user", "proposal"),), CONTEXT, 1)
                self.assertEqual(candidate.freeze().id, BASE.id)

    def test_unregistered_shell_or_check_fails_closed(self):
        for request in (tool("shell", command="echo bad"), tool("check", name="unregistered")):
            runtime = FakeNativeRuntime(NativePhase(None, (request,)))
            with self.assertRaises(ProtocolError):
                ModelLocalImprover(runtime).realize(Candidate(BASE), (Message("user", "proposal"),), CONTEXT, 1)
            self.assertEqual(len(runtime.calls), 1)
            self.assertNotIn("shell", schemas(runtime))

    def test_preparation_does_not_expose_registered_checks(self):
        suite = CheckSuite({"inspect": RegisteredCheck(lambda _: CheckResult("inspect", True))})
        runtime = FakeNativeRuntime(NativePhase(None, (tool("check", name="inspect"),)))
        with self.assertRaises(ProtocolError):
            ModelLocalImprover(runtime, suite).prepare(CONTEXT)
        self.assertNotIn("check", schemas(runtime))

    def test_inquiry_checks_cannot_mutate_active_candidate(self):
        checked = []

        def bad_check(candidate):
            checked.append(candidate)
            candidate.write("memory.md", "accidental mutation")
            return CheckResult("inspect", True, "looked at candidate")

        runtime = FakeNativeRuntime(NativePhase({"content": "The check rejected its own mutation."},
                                                (tool("check", name="inspect"),)))
        local = ModelLocalImprover(runtime, CheckSuite({"inspect": RegisteredCheck(bad_check)}))
        active = Candidate(BASE)
        active.write("memory.md", "active work in progress")
        local.investigate("Inspect current draft", CONTEXT, active)
        self.assertIsNot(checked[0], active)
        self.assertEqual(checked[0].changed_paths, active.changed_paths)
        self.assertEqual(active.read("memory.md"), "active work in progress")
        self.assertEqual(BASE.files["memory.md"], "Known facts")
        self.assertFalse(result(runtime)["passed"])

    def test_inquiry_checks_without_candidate_use_published_harness_copy(self):
        seen = []

        def inspect(candidate):
            seen.append(candidate.read("memory.md"))
            return CheckResult("inspect", True, "inspected")

        runtime = FakeNativeRuntime(NativePhase({"content": "Inspected facts."}, (tool("check", name="inspect"),)))
        ModelLocalImprover(runtime, CheckSuite({"inspect": RegisteredCheck(inspect)})).investigate("Inspect", CONTEXT)
        self.assertEqual(seen, ["Known facts"])
        self.assertNotIn("candidate", schemas(runtime)["read"]["properties"]["source"]["enum"])

    def test_malformed_final_protocol_is_not_repaired_or_coerced(self):
        responses = ['```json\n{"content":"text"}\n```', '{"content":"a","content":"b"}',
                     {"content": "text", "extra": 1}, '["final"]', '{"content":NaN}',
                     {"action": "tool", "name": "read", "arguments": {}},
                     {"action": "final", "content": "legacy envelope"}, {"content": ""}]
        for reply in responses:
            with self.subTest(reply=reply):
                runtime = FakeNativeRuntime(reply)
                with self.assertRaises(ProtocolError):
                    ModelLocalImprover(runtime).prepare(CONTEXT)
                self.assertEqual(len(runtime.calls), 1)

    def test_clarification_keeps_candidate_and_explicit_phase_result(self):
        runtime = FakeNativeRuntime(
            NativePhase({"status": "CLARIFY", "clarification": "Use the fallback?"},
                        (tool("write", path="memory.md", content="intermediate"),)),
            NativePhase({"status": "STOP", "review": "Proposal withdrawn."},
                        (tool("read", source="candidate", path="memory.md"),)))
        local = ModelLocalImprover(runtime)
        candidate = Candidate(BASE)
        first = local.realize(candidate, (Message("user", "proposal"),), CONTEXT, 3)
        second = local.realize(candidate, (Message("user", "proposal"), Message("assistant", "withdraw")), CONTEXT, 2)
        self.assertEqual((first.status, second.status), ("CLARIFY", "STOP"))
        self.assertEqual(result(runtime)["content"], "intermediate")
        self.assertIn("Use the fallback?", runtime.calls[-1]["prompt"])
        self.assertEqual(candidate.read("memory.md"), "intermediate")

    def test_realization_boolean_status_and_clarification_budget_are_enforced(self):
        cases = [({"status": "READY", "reviewed": "true", "review": "review"}, 1),
                 ({"status": "CLARIFY", "clarification": "question"}, 0),
                 ({"status": "READY", "reviewed": True, "review": "review", "extra": 1}, 1),
                 ({"status": "UNKNOWN", "review": "review"}, 1)]
        for reply, remaining in cases:
            with self.subTest(reply=reply):
                runtime = FakeNativeRuntime(reply)
                with self.assertRaises(ProtocolError):
                    ModelLocalImprover(runtime).realize(Candidate(BASE), (Message("user", "proposal"),), CONTEXT, remaining)

    def test_native_runtime_owns_tool_iteration_without_custom_role_step_cap(self):
        def native_phase(dispatch):
            for _ in range(100):
                dispatch("list", {"source": "harness"})
            return {"content": "Native session completed."}

        runtime = FakeNativeRuntime(native_phase)
        ModelLocalImprover(runtime).prepare(CONTEXT)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(len(runtime.calls[0]["results"]), 100)

    def test_long_experience_initial_overview_is_bounded_and_tail_remains_accessible(self):
        experience = Experience("long", "Preserved task description", "outcome",
                                ("prefix" * 40000, "LATE_EXPERIENCE_SENTINEL"), "feedback")
        serialized = json.dumps(experience.to_dict(), ensure_ascii=False)
        offset = serialized.index("LATE_EXPERIENCE_SENTINEL")
        runtime = FakeNativeRuntime(NativePhase({"content": "Inspected late observation."},
            (tool("read", source="experience", offset=offset, limit=100),)),
            settings=replace(ModelSettings.local(), tool_output_tokens=512))
        ModelLocalImprover(runtime).prepare(LocalContext(BASE, experience, (), {}))
        prompt = runtime.calls[0]["prompt"]
        self.assertIn("Preserved task description", prompt)
        self.assertNotIn("prefix" * 100, prompt)
        self.assertIn("full_record_access", prompt)
        page = result(runtime)
        self.assertIn("LATE_EXPERIENCE_SENTINEL", page["content"])
        self.assertEqual((page["offset"], page["total_chars"], page["format"]), (offset, len(serialized), "json"))

    def test_all_local_read_sources_reach_content_beyond_preview(self):
        value = "prefix" * 10000 + "LATE_CONTENT_SENTINEL"
        harness = Harness({**BASE.files, "skills/large/SKILL.md": value})
        historical = Experience("old", "task", value)
        history_json = json.dumps(historical.to_dict(), ensure_ascii=False)
        context = LocalContext(harness, CONTEXT.experience, (historical,), {"large": value})
        sources = [("harness", "skills/large/SKILL.md", value), ("candidate", "skills/large/SKILL.md", value),
                   ("resource", "large", value), ("history", "old", history_json)]
        for source, path, complete_text in sources:
            with self.subTest(source=source):
                runtime = FakeNativeRuntime(NativePhase({"content": "Read late evidence."}, (
                    tool("read", source=source, path=path, offset=complete_text.index("LATE_CONTENT_SENTINEL"), limit=100),)),
                    settings=replace(ModelSettings.local(), tool_output_tokens=256))
                ModelLocalImprover(runtime).investigate("Read tail", context, Candidate(harness))
                page = result(runtime)
                self.assertIn("LATE_CONTENT_SENTINEL", page["content"])
                self.assertEqual(page["total_chars"], len(complete_text))
                encoded = runtime.calls[0]["results"][0]["output"]
                self.assertLessEqual(conservative_token_count(encoded), 256)
                self.assertNotIn("truncated", json.loads(encoded))

    def test_default_pages_preserve_continuation_and_reconstruct_exact_private_content(self):
        value = ('前缀\\"\n' * 200) + "LATE_SENTINEL"
        segments = []

        def native_phase(dispatch):
            args = {"source": "resource", "path": "paged"}
            while True:
                page = json.loads(dispatch("read", args))["result"]
                segments.append(page["content"])
                if page["next_offset"] is None:
                    return {"content": "Read entire private resource."}
                args["offset"] = page["next_offset"]

        runtime = FakeNativeRuntime(native_phase, settings=replace(ModelSettings.local(), tool_output_tokens=384))
        ModelLocalImprover(runtime).prepare(LocalContext(BASE, CONTEXT.experience, (), {"paged": value}))
        self.assertEqual("".join(segments), value)
        self.assertGreater(len(segments), 1)
        for item in runtime.calls[0]["results"]:
            self.assertLessEqual(conservative_token_count(item["output"]), 384)
            self.assertNotIn("truncated", json.loads(item["output"]))

    def test_private_source_lists_are_pageable_json(self):
        resources = {f"resource-{i:03}": "private value" for i in range(100)}
        segments = []

        def native_phase(dispatch):
            offset = 0
            while offset is not None:
                page = json.loads(dispatch("list", {"source": "resource", "offset": offset}))["result"]
                segments.append(page["content"])
                offset = page["next_offset"]
            return {"content": "Listed resources."}

        runtime = FakeNativeRuntime(native_phase, settings=replace(ModelSettings.local(), tool_output_tokens=256))
        ModelLocalImprover(runtime).prepare(LocalContext(BASE, CONTEXT.experience, (), resources))
        self.assertEqual(json.loads("".join(segments)), sorted(resources))

    def test_invalid_page_parameters_fail_without_another_native_phase(self):
        for paging in ({"offset": -1}, {"offset": True}, {"offset": "1"}, {"limit": 0}, {"limit": 65537},
                       {"limit": False}, {"limit": 1.5}, {"offset": 10000, "limit": 10}):
            with self.subTest(paging=paging):
                runtime = FakeNativeRuntime(NativePhase(None, (tool("read", source="harness", path="memory.md", **paging),)))
                with self.assertRaises(ProtocolError):
                    ModelLocalImprover(runtime).prepare(CONTEXT)
                self.assertEqual(len(runtime.calls), 1)

    def test_too_small_page_budget_rejects_instead_of_losing_continuation(self):
        runtime = FakeNativeRuntime(NativePhase(None, (tool("read", source="harness", path="memory.md"),)),
                                    settings=replace(ModelSettings.local(), tool_output_tokens=32))
        with self.assertRaises(ProtocolError):
            ModelLocalImprover(runtime).prepare(CONTEXT)


class CloudAgentTests(unittest.TestCase):
    def test_identical_briefs_never_reuse_hidden_answers(self):
        runtime = FakeNativeRuntime({"action": "PROPOSE", "content": "old round unique proposal"},
                                    {"action": "RETAIN", "content": "fresh decision"})
        cloud = ModelCloudImprover(runtime)
        same_brief = (Message("user", "identical summary"),)
        cloud.decide(same_brief, 3)
        cloud.decide(same_brief, 3)
        self.assertNotIn("old round unique proposal", runtime.calls[-1]["prompt"])
        self.assertEqual(runtime.calls[0], runtime.calls[1])

    def test_complete_authoritative_transcript_has_no_duplication_or_compaction(self):
        runtime = FakeNativeRuntime({"action": "ASK", "content": "unique question"},
                                    {"action": "PROPOSE", "content": "proposal"},
                                    settings=ModelSettings("cloud", context_tokens=1000, compact_at=800, max_output_tokens=100))
        cloud = ModelCloudImprover(runtime)
        initial = (Message("user", "summary"),)
        cloud.decide(initial, 3)
        supplied = initial + (Message("assistant", "unique question"), Message("user", "released answer " * 1000))
        cloud.decide(supplied, 2)
        prompt = runtime.calls[-1]["prompt"]
        self.assertEqual(prompt.count("unique question"), 1)
        self.assertEqual(json.loads(prompt)["transcript"], [{"role": m.role, "content": m.content} for m in supplied])
        self.assertEqual(runtime.calls[-1]["tools"], [])

    def test_cloud_has_only_guarded_input_and_correct_phase_actions(self):
        runtime = FakeNativeRuntime({"action": "ASK", "content": "Which behavior failed?"},
                                    {"action": "PROPOSE", "content": "Update memory."},
                                    {"action": "RESPOND", "content": "Use documented fallback."})
        cloud = ModelCloudImprover(runtime)
        initial = (Message("user", "BLOCKED"),)
        self.assertEqual(cloud.decide(initial, 3).action, "ASK")
        supplied = initial + (Message("assistant", "Which behavior failed?"), Message("user", "sanitized evidence"))
        self.assertEqual(cloud.decide(supplied, 2).action, "PROPOSE")
        self.assertEqual(cloud.respond(supplied + (Message("user", "clarification"),), 2).action, "RESPOND")
        prompt = runtime.calls[-1]["prompt"]
        self.assertIn("BLOCKED", prompt)
        self.assertIn("sanitized evidence", prompt)
        self.assertNotIn("private resource contents", prompt)
        self.assertNotIn("private task", prompt)
        self.assertEqual(runtime.calls[-1]["tools"], [])

    def test_cloud_dispatch_fails_closed_even_when_runtime_attempts_tool(self):
        runtime = FakeNativeRuntime(NativePhase(None, (tool("read", source="resource", path="manual"),)))
        with self.assertRaises(ProtocolError):
            ModelCloudImprover(runtime).decide((Message("user", "brief"),), 1)

    def test_cloud_enforces_phase_actions_fields_and_inquiry_budget(self):
        cases = [("decide", {"action": "RESPOND", "content": "answer"}, 1),
                 ("respond", {"action": "RETAIN", "content": "retain"}, 1),
                 ("decide", {"action": "ASK", "content": "question"}, 0),
                 ("decide", {"action": "PROPOSE", "content": "proposal", "private": "data"}, 1)]
        for phase, reply, budget in cases:
            with self.subTest(phase=phase, reply=reply):
                runtime = FakeNativeRuntime(reply)
                with self.assertRaises(ProtocolError):
                    getattr(ModelCloudImprover(runtime), phase)((Message("user", "summary"),), budget)
                self.assertEqual(len(runtime.calls), 1)


class WorkerTests(unittest.TestCase):
    def test_worker_loads_context_and_exposes_signature_and_docstring_native_tool(self):
        def lookup(key: str, count: int = 1):
            """Look up an authorized fixture key."""
            return {"value": ("fixture:" + key) * count}

        runtime = FakeNativeRuntime(NativePhase({"content": "task answer"}, (
            tool("read_harness", path="memory.md"), tool("lookup", key="sample"))))
        worker = ModelWorker(runtime, {"lookup": lookup})
        experience = worker.execute(Task("task1", "Answer task"), BASE)
        self.assertEqual(experience.outcome, "task answer")
        self.assertEqual(experience.feedback, "")
        self.assertEqual(len(experience.observations), 2)
        self.assertIn("fixture:sample", experience.observations[-1])
        self.assertIn("Follow relevant skills.", runtime.calls[0]["prompt"])
        self.assertEqual(BASE.files["memory.md"], "Known facts")
        schema = schemas(runtime)["lookup"]
        self.assertEqual(schema["required"], ["key"])
        self.assertEqual(schema["properties"]["key"], {"type": "string"})
        self.assertEqual(schema["properties"]["count"], {"type": "integer", "default": 1})
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("authorized fixture", runtime.calls[0]["tools"][-1]["description"])
        self.assertEqual(set(schemas(runtime)), {"read_harness", "list_harness", "lookup"})

    def test_explicit_host_tool_schema_is_registered(self):
        def lookup(**arguments):
            return arguments["item"]

        schema = {"type": "object", "properties": {"item": {"enum": ["yes", "no"]}}, "required": ["item"]}
        runtime = FakeNativeRuntime(NativePhase({"content": "yes"}, (tool("choose", item="yes"),)))
        worker = ModelWorker(runtime, {"choose": HostTool(lookup, schema, "Choose the supplied item.")})
        worker.execute(Task("task", "Choose"), BASE)
        self.assertEqual(schemas(runtime)["choose"], schema)
        self.assertEqual(result(runtime), "yes")

    def test_worker_cannot_call_implicit_shell_write_or_check(self):
        for name in ("shell", "write", "check"):
            runtime = FakeNativeRuntime(NativePhase(None, (tool(name),)))
            with self.assertRaises(ProtocolError):
                ModelWorker(runtime).execute(Task("task1", "Run task"), BASE)
            self.assertNotIn(name, schemas(runtime))
            self.assertEqual(len(runtime.calls), 1)

    def test_worker_reload_context_and_start_each_task_without_prior_task_evidence(self):
        runtime = FakeNativeRuntime(NativePhase({"content": "previous private answer"},
                                                (tool("private"),)), {"content": "next answer"})
        worker = ModelWorker(runtime, {"private": lambda: "previous private evidence"})
        worker.execute(Task("first", "previous private task"), BASE)
        updated = Harness({**BASE.files, "CONTEXT.md": "Updated task policy."})
        second = worker.execute(Task("second", "new task"), updated)
        self.assertEqual(second.observations, ())
        self.assertIn("Updated task policy.", runtime.calls[1]["prompt"])
        self.assertNotIn("previous private", runtime.calls[1]["prompt"])

    def test_worker_read_harness_pages_to_tail(self):
        prefix = "large artifact prefix " * 10000
        harness = Harness({**BASE.files, "skills/large/SKILL.md": prefix + "LATE_WORKER_SENTINEL"})
        runtime = FakeNativeRuntime(NativePhase({"content": "Found late artifact."}, (
            tool("read_harness", path="skills/large/SKILL.md", offset=len(prefix), limit=100),)),
            settings=replace(ModelSettings.local(), tool_output_tokens=256))
        experience = ModelWorker(runtime).execute(Task("t", "Inspect artifact tail"), harness)
        page = json.loads(experience.observations[0])["result"]
        self.assertEqual(page["content"], "LATE_WORKER_SENTINEL")
        self.assertIsNone(page["next_offset"])
        self.assertLessEqual(conservative_token_count(experience.observations[0]), 256)

    def test_worker_large_host_outputs_are_bounded_in_native_results_and_observations(self):
        runtime = FakeNativeRuntime(NativePhase({"content": "answer"}, (tool("large"),)),
                                    settings=replace(ModelSettings.local(), tool_output_tokens=256))
        experience = ModelWorker(runtime, {"large": lambda: "evidence" * 1000}).execute(Task("t", "task"), BASE)
        self.assertEqual(experience.observations[0], runtime.calls[0]["results"][0]["output"])
        self.assertTrue(json.loads(experience.observations[0])["truncated"])
        self.assertLessEqual(conservative_token_count(experience.observations[0]), 256)

    def test_worker_rejects_invalid_registered_arguments_before_callback(self):
        invoked = []

        def lookup(key):
            invoked.append(key)

        runtime = FakeNativeRuntime(NativePhase(None, (tool("lookup", wrong="bad"),)))
        with self.assertRaises(ProtocolError):
            ModelWorker(runtime, {"lookup": lookup}).execute(Task("t", "task"), BASE)
        self.assertEqual(invoked, [])

    def test_worker_requires_strict_final_content_without_legacy_actions(self):
        for final in ("raw answer", {"content": "answer", "action": "final"}, {"content": ""}):
            runtime = FakeNativeRuntime(final)
            with self.assertRaises(ProtocolError):
                ModelWorker(runtime).execute(Task("t", "task"), BASE)
            self.assertEqual(len(runtime.calls), 1)

    def test_worker_reserved_tool_names_cannot_be_replaced(self):
        for name in ("read_harness", "list_harness", "", " "):
            with self.assertRaises(ValueError):
                ModelWorker(FakeNativeRuntime(), {name: lambda: None})


if __name__ == "__main__":
    unittest.main()
