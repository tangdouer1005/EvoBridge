# Method and implementation

EvoBridge coordinates a Local Worker, a Local Improver, and a Cloud Improver.
The runtime implements the two interaction loops in Sections 3.1–3.4 and
Algorithm 1 of the accompanying paper.

## Components

| Component | Implementation | Paper reference |
| --- | --- | --- |
| Persistent harness | `Harness` stores domain memory in `memory.md`, reusable operations in `skills/`, and execution policy in `CONTEXT.md`. | §3.1 |
| Task execution | The Worker loads the current policy at task initialization and uses its fixed tools and model. | §3.1 |
| Local preparation | The Local Improver summarizes outcomes, observations, capability use, and unresolved questions, distinguishing facts from hypotheses. | §3.2 |
| Cloud inquiry | `ASK`, `PROPOSE`, and `RETAIN` select investigation, an update proposal, or the existing harness. | §3.2 |
| Update design | Proposals describe affected M/O/π components, intended behavior, applicability, inputs/outputs, preserved behavior, and expected checks. Private bindings are resolved locally. | §3.3 |
| Local realization | Edits are applied to a candidate copy. Executable operations require behavioral checks; memory and policy edits require an applicability, consistency, and capability-link review. | §3.3 |
| Clarification | The Local Improver reports a design ambiguity, unavailable capability, or observed mismatch. The cloud clarifies, revises, withdraws, or requests follow-up investigation. | §3.3 |
| Publication | `READY`, a completed review, and passing required checks publish the candidate's actual file changes. Withdrawal, `STOP`, or an unfinished candidate at the limit retains the previous harness. | §3.4, Algorithm 1 |

Evolution changes harness artifacts while keeping model weights, the Worker's
execution loop, and access permissions fixed. Both successful and unsuccessful
task experiences can inform an update. Publication depends on artifact review
and local checks, rather than improvement on an evaluation score.

## Shared transcript and Policy Guard

The built-in cloud adapter receives role instructions and a guarded transcript.
Raw execution records, resource mappings, and harness files are not directly
attached to cloud requests. The transcript spans inquiry and realization.

`PolicyGuard` applies configured prohibited regex patterns to each outgoing
local message. A match replaces the entire message with the fixed string
`BLOCKED`; an unmatched message is forwarded unchanged. This applies to initial
briefs, investigation answers, clarification requests, and follow-up answers.

The rule coverage is deployment-specific. An unmatched message can still contain
sensitive information, so the guard does not establish a general privacy
guarantee. Patterns should reflect the application's sharing policy. The adapter
does not attach raw evidence files or local audit logs to cloud requests.

## Interaction budgets

The default exchange limits are `BI = BR = 3` (Appendix B.3).

- **Inquiry:** at most `BI` question/answer exchanges, followed when necessary
  by a final proposal/retain decision with zero remaining inquiries. This gives
  at most `BI + 1` cloud decisions.
- **Realization:** at most `BR` clarification responses and `BR + 1` local
  realization attempts. A valid `READY` on the final attempt can publish.
- **Zero limits:** one initial cloud decision and one local realization attempt
  remain available, without inquiry or clarification exchanges.
- **Follow-up investigation:** an implementation-stage `ASK` uses the remaining
  global inquiry allowance, and each cloud clarification response uses the
  realization allowance. This is the runtime's accounting convention for
  follow-up investigations described in §3.3.

## Execution schedules

**Online.** The Worker executes task `t` with `h_t`. Its original outcome and
authorized feedback are recorded before evolution; published changes apply to
subsequent tasks. Investigation and local checks preserve that recorded outcome.
Reset-10 restores the initial harness and clears accessible history between
ten-task blocks.

**Offline.** The engine revisits a fixed corpus of recorded trajectories and
authorized feedback. Source tasks are not re-executed. Default checkpoints are
epochs `0, 1, 2, 3, 5`.

**Frozen evaluation and transfer.** The Worker uses a fixed harness without
evolution or cloud improvement. The caller supplies target environments with
their original tools and permissions. Evaluation feedback is excluded from
adaptation.

## Model profiles

The example profiles follow Appendix B.3:

| Setting | Qwen3.5-9B | Qwen3.8-27B |
| --- | ---: | ---: |
| Thinking | Disabled | Disabled |
| Temperature | 0.6 | 1.0 |
| Top-p / top-k | 0.95 / 20 | 0.95 / 20 |
| Min-p / presence penalty | 0 / 0 | 0 / 0 |
| Repetition penalty | 1.0 | 1.0 |
| Runtime context | 131,072 | 131,072 |
| Compaction threshold | 98,304 | 98,304 |
| Maximum output | 32,768 | 32,768 |
| Per-tool output limit | 8,192 | 8,192 |
| Native model context | 262,144 | 262,144 |
| Weight precision | BF16 | BF16 |
| KV cache precision | BF16 / FP16 | BF16 / FP16 |

Token limits are in tokens. Worker and Local Improver share the selected local
profile. The cloud profile uses `gpt6-sol` with `medium` reasoning. Checkpoint
identity, native context, and precision are serving-side settings; the API client
configures sampling and runtime limits.

## Runtime conventions

The implementation uses strict JSON role responses, immutable snapshots,
candidate copies, and durable task records. Deployment configuration supplies
privacy patterns and local checks. Non-documentation changes under `skills/`
require at least one registered behavioral check; the application is responsible
for covering the affected operations.

Prompts, check interfaces, context compaction, and interruption handling are
specified in the [integration guide](integration.md). The configurable per-phase
tool-step limit is separate from the paper's interaction budgets. Framework tests
and synthetic examples validate these implementation contracts; benchmark
experiments are outside this source release.
