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
registry. Private binding and execution stay local. Tools are registered with
native schemas and host callbacks; `agents.py` parses only the completed role
outputs. Malformed final JSON stops the round without publishing partial files.

## Execution and model serving

Model-backed roles use the installed agent runtime through its stdio app-server
interface. The tested dependency is Codex CLI 0.162.1; `--runtime-executable` can
select its executable. The [runtime protocol documentation](https://developers.openai.com/codex/app-server)
provides the upstream API reference. If the executable is missing or its protocol
is incompatible, execution stops; there is no fallback agent loop.

Each role phase starts an isolated ephemeral session. The runtime owns model
turns, native tool dispatch, and automatic compaction. EvoBridge registers only
the phase's permitted dynamic tools. Shell, browser, plugin, memory, and other
host integrations are disabled. Unknown capabilities and approval requests are
rejected. The cloud session has no tools, and receives only the guarded
transcript. Local investigation and realization receive explicit private
resources through scoped tools. User-level configuration and login files are
not inherited by these sessions.

The configured endpoints must support the Responses API, including native
function-call items and streaming responses. An authenticated loopback proxy
forwards these requests, enforcing the configured model and permitted tools.
It supplies provider credentials from the named environment variable and keeps
the upstream credential out of the runtime subprocess. It does not convert Chat
Completions responses or rewrite generated model/tool output.

Regular `/responses` requests use the model profile's sampling and output limit.
Local requests explicitly set `reasoning.effort=none`; cloud requests set
`reasoning.effort=medium`. Local sampling includes `temperature`, `top_p`,
`top_k`, `min_p`, `presence_penalty`, and `repetition_penalty`; the local serving
endpoint must support these parameters. Requests to `/responses/compact`, when
used by the runtime, preserve that endpoint's narrower schema and fix the model.
Server configuration establishes the checkpoint and numeric precision.

The local context window is 131,072 tokens, compaction begins at 98,304, maximum
output is 32,768 tokens, and tool output is limited to 8,192 tokens. These values
are supplied to the runtime or provider as appropriate. Application pagination
also uses a conservative UTF-8 byte estimate to bound tool envelopes; it is
separate from native tokenization, compaction, and provider-reported usage.
The `BI`/`BR` budgets bound collaboration exchanges, not the native execution
loop's internal tool calls.

Only connection establishment and initial session admission are time-bounded.
Accepted generation has no additional wall-clock deadline. Request and stream
retries are disabled; a failed turn is not silently replayed. Interrupted or
incomplete generations cannot be accepted as completed role outputs.

Provider receipts count individual upstream requests and preserve missing usage
as `None`. Cached-input and reasoning-output counts are subsets of the reported
input and output totals. Native session receipts separately report turn status,
compaction events, and runtime token totals; session totals must not be added to
provider totals or treated as provider request counts. Failed requests may incur
charges. The offline CLI writes both kinds of records to the private store;
abrupt process termination can leave accounting incomplete.

## Resume and failure semantics

Use a fresh private work directory per independent run. Initialization freezes
the run configuration, runtime backend/endpoint, policy, initial harness,
resources, and named check set.
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
