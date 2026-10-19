# Integration and runtime details

## Application boundary

`Task` has a stable id and instruction. A `Worker.execute(task, harness)` adapter
returns an `Experience` containing the same id, its original outcome, private
observations, and available authorized feedback. `Runner` durably records this
primary before calling `Engine.evolve`. It never revises the stored outcome using
an investigation, check, or later execution. Model-generated Worker answers are
not automatically graded.

`ModelWorker` loads `CONTEXT.md` at the start of each task. Harness read/list tools
give it access to `memory.md` and `skills/`; environment actions must be explicitly
registered by the caller. The tool set and model client remain outside the
editable harness. A callable tool receives model-supplied keyword arguments and
must validate authorization and resource isolation itself.

`LocalContext` contains the current immutable harness, completed experience,
accessible history, and a mapping of private resource names to text. The model
Local Improver can read those resources and edit only its candidate. There is no
automatic host-filesystem mount, shell execution, model training, or benchmark
environment. Register a `Worker` or custom `LocalImprover` when your application
needs additional controlled tools.

Private reads support character-based `offset` and `limit` pagination. A page
includes `next_offset` and is shortened as needed to fit its complete tool-result
envelope. Follow that continuation until it is `null`; a preview is not the full
record. Preparation pins only a bounded overview of long experience, leaving the
raw record available through paged reads. The Worker can likewise page long
harness artifacts without gaining any additional environment tool permissions.

## Checks and actual artifacts

`CheckSuite` registers named trusted callables returning `CheckResult`. All
registered checks are required before publication; executable-artifact changes
also require at least one check marked `behavioral=True`. Syntax validation alone
is not a behavioral test. The host should cover every affected operation and its
intended inputs, outputs, and preserved behavior.

The Local Improver can run checks while working, and the runtime repeats them on
the final candidate after `READY`. Each check receives a detached copy preserving
the original base and diff. Results are tied to the actual candidate content
hash. A check that modifies its copy fails without changing the working candidate. The Local
Improver must also provide an artifact review. Review is an agent judgment, not
proof of correctness. Adoption checks exclude held-out evaluation feedback and
official task-score comparisons.

The artifact API supports additions, edits, and deletion of operation files.
`memory.md` and `CONTEXT.md` remain present (they may be empty). Symlinks,
hardlinks, traversal, noncanonical paths, binary contents, and edits outside
these documents and `skills/` are rejected. Import/export is UTF-8 text only.

## Cloud request boundary

The engine calls the cloud with `Message` objects from its guarded transcript.
Only `PolicyGuard.release` adds new local content to that transcript. A match
replaces an entire draft with exactly `BLOCKED`; the cloud receives neither
matched substrings nor private resource names from audit metadata. The initial
brief, inquiry answers, clarification requests, and follow-up answers all use
this path. Local event logs contain private drafts for diagnosis and must remain
local. They are not part of the shared transcript.

Cloud actions have a strict JSON envelope:

```json
{"action": "PROPOSE", "content": "Describe targets, intended behavior, scope, and expected outcomes."}
```

Inquiry supports `ASK`, `PROPOSE`, `RETAIN`; implementation responses support
`RESPOND`, `PROPOSE`, `ASK`, `WITHDRAW`. The cloud has no file or environment tool
registry. Private binding and execution stay local. See `agents.py` for the local
tool/final envelopes. Invalid JSON or unavailable actions stop that round without
publishing partial files; there is no model-output repair/retry loop.

## Model serving and token accounting

The built-in transport uses `/chat/completions`, JSON request/response bodies,
and an environment-variable bearer credential when configured. The server must
support the requested parameters. Local sampling includes `top_k`, `min_p`,
`repetition_penalty`, and `chat_template_kwargs.enable_thinking=false`. Cloud
requests include `reasoning_effort=medium`. Unsupported server fields cause a
failure rather than silent removal. No model-name or endpoint substitution is
performed. Model aliases and BF16 deployment remain the operator's responsibility.

The paper's local limits are 131,072 runtime-context tokens, compaction at 98,304,
32,768 maximum output tokens, and 8,192 per tool output. A caller can inject the
deployed model's token counter into `HTTPChatClient`. Without one, the framework
uses UTF-8 byte length plus framing overhead as a conservative estimate, not an
exact tokenizer. Server-reported usage remains separate from estimates.

Compaction removes older conversational tool exchanges with an explicit omission
notice, retaining essential task and phase data without generating summaries. If required context cannot fit alongside the output reservation,
the request fails before sending. This compaction policy is an implementation
choice: the paper specifies the threshold, not the algorithm. Full guarded cloud
transcripts must fit; they are not truncated to make a new request.

The built-in agent tool loop has a configurable default cap of 64 tool actions
per phase. This is a separate engineering limit from `BI`/`BR`, not an additional
paper hyperparameter. All phases use the same local model settings. Configure
these tool limits on custom agents as needed; existing task tools/permissions
must stay fixed for a comparison.

Usage receipts distinguish attempted transport calls from local preflight
failures. Missing token counts are `None`, never zero. Cached-input and reasoning
output counts, when supplied by the server, are subsets of the reported input
and output totals and must not be added again. There are no automatic request
retries. Failed requests may incur provider charges. The offline CLI saves usage
receipts to the private store when the invocation exits normally or raises a
caught exception; abrupt process termination can leave usage unavailable.

## Resume and failure semantics

Use a fresh private work directory per independent run. Initialization freezes
the run configuration, policy, initial harness, resources, and named check set.
The application must also freeze actual tool/check implementations and serving
configuration; their names alone cannot attest to remote behavior.

Completed rounds and primary tasks are idempotent. Reusing an id with changed
inputs is rejected. Atomic start claims prevent two processes from dispatching
the same call. Changing task order, source corpus, or schedule mode in an existing
directory is rejected. Changing `epochs` requires a new offline run directory;
stored checkpoint reconstruction never reruns completed source tasks.

An interrupted start with no completion raises `PendingRound`; it is never
automatically resubmitted. Inspect the private records and provider state before
deciding how to reconcile it. This release does not include an automated
reconciliation command. A publication/completion I/O failure is also treated as
pending because the snapshot pointer may already have advanced. It is not
recorded as a failed old-harness result. Do not delete start records merely to
force a retry of an uncertain call.

For parallel experiments, use independent engines/stores per group. Share model
serving capacity through a deployment scheduler if needed; one online sequence
remains sequential. Host-created callbacks are trusted code, so process and
network sandboxing belongs in the application adapter, not this Python library.
