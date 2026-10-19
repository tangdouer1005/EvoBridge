# Alignment with the final EvoBridge paper

This repository implements the benchmark-independent method in Sections 3.1–3.4 and Algorithm 1 of the final EvoBridge paper. The references below use the paper's printed page and manuscript line numbers. The private paper PDF is not distributed in this repository.

## Method contract

| Paper requirement | Repository contract | Reference |
| --- | --- | --- |
| Persistent harness `h = (M, O, π)` | `memory.md` stores domain facts, rules and applicability conditions; `skills/` stores reusable operations; `CONTEXT.md` guides knowledge loading, operation selection and workflow composition. The Worker receives the current harness for each task. | p. 3, lines 205–219 |
| Three distinct roles | The local Worker executes tasks. The Local Improver prepares experience, investigates, edits and checks. The Cloud Improver requests evidence and designs updates from the shared transcript. Worker and Local Improver use the same local model checkpoint. | p. 3, lines 220–231 |
| Fixed execution capability | Evolution changes harness artifacts. It does not change model weights, the Worker's core execution loop, tools or access permissions. | p. 3, lines 225–227; p. 14, lines 1091–1095 |
| Local experience preparation | The brief describes the completed outcome, important actions and observations, capability use and unresolved questions. Facts and tentative explanations are distinguished. Successful and unsuccessful experiences are both eligible. | p. 3, lines 249–262 |
| Cloud-directed inquiry | The Cloud Improver chooses `ASK`, `PROPOSE` or `RETAIN`. Investigation uses local history, resources and the harness, returning findings, supporting observations and remaining uncertainty. | p. 4, lines 263–282 |
| Concrete update design | Proposals identify affected memory, operations or policy, intended behavior, applicability, behavior to preserve, inputs, outputs, main steps and expected verification outcomes. Private names, source locations and interface bindings are resolved locally. Existing matching capabilities are revised rather than duplicated. | p. 4, lines 287–305 |
| Local realization | The Local Improver works on a candidate copy and resolves routine implementation choices locally. Executable operations require focused behavior and input/output checks; memory and policy changes require applicability, consistency and capability-link review. | p. 4, lines 306–317 |
| Design clarification | Consequential ambiguity, unavailable capability or conflict with observed behavior can trigger clarification. A request reports the local condition, needed decision and check results. The cloud can clarify, revise, withdraw or request investigation; local edits and checks then resume. | p. 5, lines 318–333 |
| Adoption | `READY` requires a completed artifact review and passing required local checks. Publication adopts the candidate's actual file changes for subsequent tasks. There is no additional task-success, reward-improvement, held-out-score or benchmark-score gate. | p. 5, lines 341–349; Algorithm 1, steps 18–22 |
| Retention | `RETAIN`, a withdrawn proposal, `STOP` or an unfinished candidate at the interaction limit preserves the previous harness. Unpublished candidate edits are not used by later tasks. | p. 5, lines 345–347 |

## Information boundary

Execution history, private resources, source files and complete editable harness artifacts remain local. Cloud calls receive only the role instructions and shared transcript. The transcript preserves context across information acquisition and realization.

`PolicyGuard` implements the paper's exact message-level rule: if any configured prohibited regex matches an outgoing message, the **entire message** becomes the fixed string `BLOCKED`; otherwise the message is forwarded unchanged. This is not substring redaction. The boundary covers the initial experience brief, investigation answers and implementation-stage clarification requests, including follow-up investigation evidence. Cloud-authored messages may be appended directly because they contain no new local disclosure. See p. 3, lines 232–245 and p. 4, lines 271–274.

The paper does not provide the concrete prohibited-pattern set. Patterns are deployment configuration, not a comprehensive privacy guarantee. Its stated limitation is that protection depends on configured rule coverage; contextual disclosure constraints require richer mechanisms (p. 9, lines 615–620). Raw local evidence must not be added to cloud system prompts, metadata, tool results, retry messages or context-compaction requests outside this boundary.

## Interaction budgets and edge cases

The paper sets `BI = BR = 3` and describes both as maximum exchange counts (p. 5, lines 339–340; p. 14, lines 1096–1102).

- **Inquiry:** up to `BI` question/answer exchanges. There can be `BI + 1` cloud decisions because the final decision at zero remaining budget must propose or retain. Another `ASK` at zero is invalid and cannot trigger investigation.
- **Realization:** up to `BR` cloud clarification responses and `BR + 1` local realization attempts. On the final attempt, `READY` is checked before exhaustion, so a completed valid candidate can still publish. An unfinished candidate cannot trigger another cloud response.
- **Zero budgets:** the cloud can still make its initial proposal/retain decision and the Local Improver can still attempt realization. No inquiry or clarification exchange is allowed.
- **Follow-up investigation:** the paper permits cloud-requested investigation during implementation but does not give a separate accounting rule (p. 5, lines 324–328). This implementation uses the same global `BI` investigation allowance and charges each cloud clarification response against the shared `BR` allowance. An implementation-stage `ASK` therefore consumes an available investigation exchange as well as a clarification response. There is no unbounded third loop or reset of either allowance.

The inclusive loops in Algorithm 1 are interpreted in light of the explicit exchange caps. In particular, its `i = 0, …, BI` decision loop does not authorize `BI + 1` investigations.

## Outcomes and execution schedules

**Online:** execute task `t` using `h_t`, record its primary outcome and available feedback, and then evolve. The published harness affects subsequent tasks. Investigation, local checks and learning-only work do not replace the recorded outcome (p. 5, lines 335–349; p. 13, lines 998–1006). Scored primary execution and later adaptation are separate events; no replay can silently improve an already recorded score.

**Offline:** revisit fixed historical trajectories and authorized feedback using the same evolution engine. Source tasks need not be rerun. The paper reports checkpoints at epochs `0, 1, 2, 3, 5`; these are experiment checkpoints, not a new learning mechanism (p. 8, lines 545–552; p. 13, lines 1023–1031).

**Frozen:** execute with a fixed harness and perform no adaptation. Evaluation feedback is excluded from updates. Frozen transfer additionally disables cloud improvement while retaining the target environment, tools and permissions (p. 8, lines 569–575; p. 13, lines 1031–1036). These are generic scheduling contracts rather than integrations with the paper's benchmark environments.

## Appendix B.3 model profiles

| Setting | Qwen3.5-9B | Qwen3.8-27B |
| --- | ---: | ---: |
| Thinking | Disabled | Disabled |
| Temperature | 0.6 | 1.0 |
| Top-p | 0.95 | 0.95 |
| Top-k | 20 | 20 |
| Min-p | 0 | 0 |
| Presence penalty | 0 | 0 |
| Repetition penalty | 1.0 | 1.0 |
| Runtime context, tokens | 131,072 | 131,072 |
| Automatic compaction threshold, tokens | 98,304 | 98,304 |
| Maximum generated output, tokens | 32,768 | 32,768 |
| Per-tool output limit, tokens | 8,192 | 8,192 |
| Native model context, tokens | 262,144 | 262,144 |
| Weight precision | BF16 | BF16 |
| KV cache precision | BF16 / FP16 | BF16 / FP16 |

The cloud profile is `gpt6-sol` with `medium` reasoning. Worker and Local Improver share the selected local profile. The runtime configuration records the client-side generation and context settings; native context and numeric precision are properties of the deployment and must be configured in the model server. An API request alone cannot establish checkpoint identity or server-side precision. References: p. 14, Table 5 and lines 1082–1089, 1103–1113.

## Implementation choices and scope

Typed role responses, immutable runtime configuration, candidate isolation, durable outcome records, guarded cloud request construction and fail-closed protocol validation make the method executable and testable. These engineering mechanisms are not additional claims of algorithmic novelty. Checks are trusted, locally supplied functions. Treating non-documentation changes under `skills/` as requiring a behavioral check is a conservative executable-artifact rule; deployments can supply domain-specific checks without connecting benchmark evaluators to adoption.

The final paper does not specify exact prompts, concrete privacy patterns, artifact-review schemas, check registration, history selection, tokenization/compaction algorithms, transaction recovery or the budget accounting of follow-up inquiries. Repository choices in those areas should be understood as implementation decisions. Local review and configured checks validate the candidate; they are not a proof of correctness in every future environment.

This repository intentionally does **not** integrate ADE-bench, τ-Knowledge, EnterpriseOps-Gym, their datasets, evaluators or reference actions. It also does not reproduce baseline agents, experimental tables, task sampling, ablation runs or cross-domain result matrices. Local contract tests and example task adapters validate framework behavior; they do not establish reproduction of the paper's numerical results. The private source PDF and private execution records are not release artifacts.
