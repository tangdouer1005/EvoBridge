"""Synthetic demonstration of the two-loop interaction protocol."""
from __future__ import annotations

from .config import RunConfig
from .engine import Engine
from .harness import Harness
from .policy import PolicyGuard
from .schedules import Runner, evaluate_frozen
from .storage import SnapshotStore
from .types import CloudAction, Experience, Realization, Task


INITIAL = Harness({"memory.md": "# Domain memory\n",
                   "CONTEXT.md": "Consult memory.md and skills/ before using the existing tools.\n"})


class DemoWorker:
    model_id = "Qwen3.5-9B"

    def execute(self, task, harness):
        learned = "negative" in harness.files["memory.md"]
        return Experience(task.id, task.instruction, {"success": learned},
                          ("Synthetic refund convention check.",),
                          "Refund amounts already carry a negative sign.")


class DemoLocal:
    model_id = "Qwen3.5-9B"

    def prepare(self, context):
        return "A synthetic revenue calculation mishandled refunds; the sign convention is uncertain."

    def investigate(self, question, context, candidate=None):
        return "Observed: refunds are negative. Hypothesis: applying absolute value changed their meaning."

    def realize(self, candidate, transcript, context, remaining):
        if not any(m.role == "assistant" and '"RESPOND"' in m.content for m in transcript):
            return Realization("CLARIFY", "Should the sign rule apply only to signed ledger inputs?")
        candidate.write("memory.md", "# Domain memory\nRefunds in signed ledgers are negative.\n")
        candidate.write("skills/net-revenue.md", "For signed ledger inputs, sum the original values; do not apply abs().\n")
        candidate.write("CONTEXT.md", "Read memory.md. For signed ledger revenue tasks, consult skills/net-revenue.md.\n")
        return Realization("READY", reviewed=True,
                           review="Reviewed scope (signed ledgers), sign consistency, and policy/skill connections.")


class DemoCloud:
    model_id = "gpt6-sol"

    def decide(self, transcript, remaining):
        if len(transcript) == 1 and remaining:
            return CloudAction("ASK", "Are refunds signed, and is the current operation preserving their sign?")
        return CloudAction("PROPOSE", "Revise M/O/pi for net revenue: preserve signed amounts; bind private ledger interfaces locally.")

    def respond(self, transcript, remaining_inquiries):
        return CloudAction("RESPOND", "Yes. Scope the operation to signed ledger inputs and preserve unrelated conventions.")


def run_demo(workdir) -> dict:
    store = SnapshotStore(workdir)
    engine = Engine(store, DemoLocal(), DemoCloud(), PolicyGuard([r"PRIVATE_[A-Z0-9_]+"]),
                    config=RunConfig(), initial=INITIAL)
    results = Runner(engine, DemoWorker()).online([
        Task("demo-1", "Inspect the synthetic signed-ledger calculation."),
        Task("demo-2", "Apply the learned convention to the next synthetic task.")])
    frozen = evaluate_frozen(DemoWorker(), [Task("heldout", "Read the frozen synthetic harness.")],
                             store.current(), store=store, evaluation_id="demo-frozen")
    return {"kind": "synthetic_protocol_demo",
            "primary_outcomes": [r.outcome for r in results],
            "frozen_outcome": frozen[0].outcome,
            "snapshot": store.current().id,
            "round_statuses": [r["result"]["status"] for r in store.history("rounds")]}
