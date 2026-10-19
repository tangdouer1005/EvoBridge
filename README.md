# EvoBridge

**Evolving Enterprise Agents from Private Experience through Local–Cloud Collaboration**

EvoBridge turns completed task experience into a persistent local agent harness.
A cloud improver directs investigation and proposes updates; a local improver
investigates private evidence, implements the proposal, and asks for clarification
when local conditions affect the design. A Policy Guard screens every outgoing
local message. Model weights and the Worker's tools and permissions do not evolve.

This repository accompanies the EvoBridge paper and provides the core runtime,
model interfaces, and online, offline, and frozen execution schedules.
Benchmark integrations, datasets, baseline implementations, and experimental
results are outside the scope of this release. See the
[method description](docs/method.md) and [integration guide](docs/integration.md).

## Quick start

Python 3.10 or newer; no third-party runtime dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
evobridge demo --workdir runtime/demo
```

The synthetic demo illustrates the interaction protocol with deterministic roles:
task execution, outcome recording, inquiry, clarification, and checked
publication. It runs without model endpoints and does not measure model
performance. Repeated invocations return the recorded results.

## Method

The only mutable artifacts are:

```text
harness/
├── memory.md       # M: domain facts, rules, and applicability
├── CONTEXT.md      # π: loading, selection, and workflow guidance
└── skills/         # O: reusable procedures, helpers, and adapters
```

1. The Local Worker executes a task with the current harness. Its original result
   is recorded before learning begins.
2. The Local Improver prepares an experience brief. The guard replaces the entire
   message with `BLOCKED` if any configured prohibited pattern matches; otherwise
   it forwards the message unchanged.
3. The cloud chooses `ASK`, `PROPOSE`, or `RETAIN`. Local investigation answers
   pass through the same guard. The default inquiry budget is `BI = 3`.
4. The Local Improver edits a candidate copy. It can request design clarification;
   guarded requests and cloud replies extend the same shared transcript. The
   default clarification budget is `BR = 3`.
5. `READY` plus an artifact review and passing required local checks publishes the
   candidate's actual changes. Withdrawal, `STOP`, or an unfinished candidate at
   the limit retains the previous harness. There is no benchmark-score adoption
   gate and no selection of the best-performing candidate.

An executable change under `skills/` requires a host-registered behavioral check.
Plain memory/policy/procedure edits require the Local Improver's applicability,
consistency, and capability-link review, plus any checks registered by the host.
Checks run against copies or caller-provided sandbox resources. The framework
does not run model-authored shell commands or automatically execute generated
programs on the host.

## Model-backed offline evolution

The HTTP adapter connects to a configured OpenAI-compatible Chat Completions
endpoint using environment-based credentials.

```bash
evobridge init --directory runtime/initial-harness

export EVOBRIDGE_LOCAL_BASE_URL='http://127.0.0.1:8000/v1'
export EVOBRIDGE_CLOUD_BASE_URL='https://your-cloud-gateway.example/v1'
# Set EVOBRIDGE_CLOUD_API_KEY privately in your shell or secret manager.
# Set EVOBRIDGE_LOCAL_API_KEY only if the local server requires it.

evobridge offline \
  --workdir runtime/offline-9b \
  --config configs/paper-qwen-9b.json \
  --policy configs/policy.example.json \
  --harness runtime/initial-harness \
  --corpus examples/experiences.jsonl \
  --epochs 3
```

The example corpus and prohibited patterns are synthetic templates. Replace them
with authorized local experience and an explicit deployment sharing policy before
using private data. Add local resources with `--resources resources.json` (a
mapping of names to text), and trusted check registration with
`--checks your_module:make_checks`. See [host check example](examples/host_checks.py).

`paper-qwen-9b.json` and `paper-qwen-27b.json` record Appendix B.3 settings: local
thinking disabled, temperatures 0.6 / 1.0, top-p 0.95, top-k 20; cloud `gpt6-sol`
with `medium` reasoning. Endpoint model names must resolve to the intended
deployment. BF16 weights, cache precision, and checkpoint identity must be
established on the serving side; client configuration cannot verify them.

The built-in clients use a strict JSON agent protocol. Local tools read private
resources, edit only candidate harness paths, and invoke only registered checks.
The cloud has no local tools and receives only guarded transcript text. Failed or
malformed model calls are recorded without automatic retries or fallback edits.
See [integration and runtime details](docs/integration.md) for tool contracts,
token accounting, context handling, and interruption behavior.

## Integrating an application

The framework defines small `Worker`, `LocalImprover`, and `CloudImprover`
protocols. The built-in model-backed roles are in `evobridge.agents`; callers may
also supply compatible agents. Application tools and feedback belong to the
local environment adapter, not the cloud role.

```python
from evobridge import Engine, PolicyGuard, Runner, SnapshotStore, Task
from evobridge.agents import ModelLocalImprover, ModelCloudImprover, ModelWorker

# local_client, cloud_client, initial_harness, checks, and run_config are
# caller-owned; use the same local model/settings for Worker and Improver.
engine = Engine(
    SnapshotStore("runtime/my-application"),
    ModelLocalImprover(local_client, checks),
    ModelCloudImprover(cloud_client),
    PolicyGuard(prohibited_patterns),
    checks, run_config, initial=initial_harness,
)
worker = ModelWorker(local_client, tools=authorized_task_tools)
results = Runner(engine, worker).online([Task("task-001", "Your task instruction")])
```

`ModelWorker` returns an answer and tool observations. Application adapters
provide evaluation results and authorized feedback in the returned `Experience`.

- `Runner.online(...)`: fixed-order, pre-update primary execution.
- `Runner.online(..., reset_every=10)`: reset harness and accessible history
  between blocks, for the paper's retention comparison.
- `offline(engine, corpus, epochs=5)`: repeated passes over existing records;
  checkpoints default to `0, 1, 2, 3, 5`; no source-task re-execution.
- `evaluate_frozen(worker, tasks, harness)`: no evolution or cloud improver.
- `transfer(sources, targets, worker_factory)`: new target environment per pair,
  fixed source harness, no target updates.

Use separate work directories for independent runs or source groups. Never put
held-out feedback in the adaptation corpus. Target environment initialization,
permissions, and tools are supplied by the caller.

## Records and privacy

Run directories contain private immutable snapshots, primary outcomes, model
usage metadata, round results, and an append-only local event log. `runtime/` is
excluded from Git; keep custom run directories outside this repository or add
their paths to `.gitignore`. State files use restrictive permissions, and
snapshots are checked by content hash. Atomic claims prevent duplicate starts;
publication checks that the current snapshot is still the candidate's base.

The rule-based guard blocks messages matching configured prohibited patterns
and forwards unmatched messages unchanged. Its coverage depends on the sharing
policy and does not provide a general privacy guarantee. Local model endpoints
must be trusted for private data; application tools and checks must enforce
their environment isolation. Store credentials in environment variables and
keep private run records outside the source release.

## Layout

| Path | Purpose |
| --- | --- |
| `src/evobridge/engine.py` | Algorithm 1 and guarded two-loop interaction |
| `src/evobridge/harness.py`, `storage.py` | M/O/π artifacts, snapshots, durable records |
| `src/evobridge/agents.py`, `providers.py` | Model-backed roles and HTTP transport |
| `src/evobridge/schedules.py` | Online, offline, frozen, and transfer schedules |
| `configs/` | Explicit paper profiles and an example sharing policy |
| `tests/` | Local contract, isolation, concurrency, and mock-provider tests |
| `docs/` | Paper mapping and integration details |

The test suite runs locally with deterministic fixtures and mocked model transports.

## Source archive

From a Git checkout, run `python scripts/export_source.py` to create
`dist/evobridge-source.zip`. The archive contains the tracked source files with
normalized ZIP metadata and excludes Git history, CI configuration, build
outputs, and run records.
