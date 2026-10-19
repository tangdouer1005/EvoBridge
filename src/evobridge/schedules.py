"""Online, fixed-corpus offline, and frozen evaluation/transfer schedules."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Iterable, Mapping, Sequence

from .engine import Engine, PendingRound, identity
from .harness import Harness
from .storage import SnapshotStore
from .types import Experience, RoundResult, Task, Worker


class Runner:
    def __init__(self, engine: Engine, worker: Worker):
        if worker.model_id != engine.config.local.model:
            raise ValueError("Worker and Local Improver must use the same fixed model")
        settings = getattr(getattr(worker, "client", None), "settings", None)
        if settings is not None and settings != engine.config.local:
            raise ValueError("Worker inference settings differ from the Local Improver profile")
        self.engine, self.worker = engine, worker

    def online(self, tasks: Iterable[Task], *, evolve: bool = True,
               reset_every: int | None = None) -> list[Experience]:
        """Seal each official result BEFORE evolution. No diagnostic replay.

        reset_every implements Reset-10 when set to 10, including exclusion of
        previous blocks from the Local Improver's accessible history.
        """
        tasks = tuple(tasks)
        if len({t.id for t in tasks}) != len(tasks):
            raise ValueError("Task ids must be unique")
        if reset_every is not None and (type(reset_every) is not int or reset_every <= 0):
            raise ValueError("reset_every must be a positive integer")
        store = self.engine.store
        store.save_record("schedule", "mode", {"mode": "online"})
        initial_record = store.read_record("schedule", "online")
        spec = {"tasks": [asdict(t) for t in tasks], "evolve": evolve, "reset_every": reset_every}
        if initial_record:
            if initial_record["spec"] != spec:
                raise ValueError("Online schedule differs from the frozen task order")
            initial = Harness(initial_record["initial"])
        else:
            initial = store.current()
            store.save_record("schedule", "online", {"spec": spec, "initial": dict(initial.files)})
        results: list[Experience] = []
        for index, task in enumerate(tasks):
            # Reset only at a new block; replaying completed schedule results
            # must not roll a current snapshot backward.
            record = store.read_record("primary", task.id)
            if record is None:
                if store.read_record("primary-starts", task.id):
                    raise PendingRound("Worker call interrupted; do not rerun its primary")
                before = store.current()
                resetting = bool(reset_every and index and index % reset_every == 0)
                harness = initial if resetting else before
                # Claim BEFORE reset: a losing concurrent runner must never
                # roll an already advanced sequence back to its initial state.
                if not store.save_record("primary-starts", task.id,
                                         {"task": asdict(task), "harness": harness.id}):
                    raise PendingRound("Another process already claimed this primary")
                if resetting:
                    from .harness import Candidate
                    reset = Candidate(before)
                    for path in set(before.files) - set(initial.files):
                        reset.delete(path)
                    for path, content in initial.files.items():
                        reset.write(path, content)
                    store.publish(reset, expected_id=before.id)
                experience = self.worker.execute(task, harness)
                if not isinstance(experience, Experience) or experience.id != task.id:
                    raise ValueError("Worker must return an Experience with the task id")
                record = {"task": asdict(task), "harness": harness.id, "experience": experience.to_dict()}
                store.save_record("primary", task.id, record)
                store.save_event("primary_recorded", {"id": task.id, "harness": harness.id})
            elif record["task"] != asdict(task):
                raise ValueError("Primary task identity changed")
            experience = Experience.from_dict(record["experience"])
            results.append(experience)
            if evolve:
                block_start = (index // reset_every) * reset_every if reset_every else 0
                self.engine.evolve(experience, round_id=f"online:{index}:{task.id}",
                                   history=tuple(results[block_start:]))
        return results


def offline(engine: Engine, corpus: Sequence[Experience], epochs: int = 5,
            checkpoints: Sequence[int] = (0, 1, 2, 3, 5)) -> dict[int, Harness]:
    """Revisit fixed authorized records, never re-execute source tasks.

    Evaluation data is not an input to this schedule. Corpus/config identity is
    frozen on first use; reentry returns stored checkpoints and resumes only
    rounds that have never started.
    """
    if type(epochs) is not int or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    if any(type(e) is not int or e < 0 for e in checkpoints):
        raise ValueError("Invalid checkpoints")
    corpus = tuple(corpus)
    if len({e.id for e in corpus}) != len(corpus):
        raise ValueError("Corpus experience ids must be unique")
    store = engine.store
    store.save_record("schedule", "mode", {"mode": "offline"})
    spec = {"corpus": [e.to_dict() for e in corpus], "epochs": epochs,
            "checkpoints": list(checkpoints)}
    store.save_record("schedule", "offline", {"identity": identity(spec)})
    saved = store.read_record("checkpoints", "0")
    if saved is None:
        store.save_record("checkpoints", "0", {"files": dict(store.current().files)})
    snapshots = {}
    if 0 in checkpoints:
        snapshots[0] = Harness(store.read_record("checkpoints", "0")["files"])
    for epoch in range(1, epochs + 1):
        saved = store.read_record("checkpoints", str(epoch))
        if saved is None:
            for index, experience in enumerate(corpus):
                engine.evolve(experience, round_id=f"offline:{epoch}:{index}:{experience.id}",
                              history=corpus)
            store.save_record("checkpoints", str(epoch), {"files": dict(store.current().files)})
        if epoch in checkpoints:
            snapshots[epoch] = Harness(store.read_record("checkpoints", str(epoch))["files"])
    return snapshots


def evaluate_frozen(worker: Worker, tasks: Iterable[Task], harness: Harness,
                    *, store: SnapshotStore | None = None,
                    evaluation_id: str = "frozen") -> list[Experience]:
    """No Engine/cloud reference: evaluation feedback cannot drive updates.

    The caller creates/resets each target environment through the Worker
    adapter. Passing a store seals evaluations and prevents silent reruns.
    """
    tasks = tuple(tasks)
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Evaluation task ids must be unique")
    if store:
        settings = getattr(getattr(worker, "client", None), "settings", None)
        store.save_record("evaluations", evaluation_id, {
            "harness": harness.id, "tasks": [asdict(t) for t in tasks], "model": worker.model_id,
            "inference_settings": asdict(settings) if is_dataclass(settings) else settings})
    results = []
    for task in tasks:
        key = identity([evaluation_id, task.id])
        saved = store.read_record("evaluation-results", key) if store else None
        if saved is None:
            if store:
                if store.read_record("evaluation-starts", key):
                    raise PendingRound("Frozen execution interrupted; do not duplicate it")
                if not store.save_record("evaluation-starts", key, {"task": task.id, "harness": harness.id}):
                    raise PendingRound("Another process already claimed this frozen execution")
            exp = worker.execute(task, harness)
            if not isinstance(exp, Experience) or exp.id != task.id:
                raise ValueError("Invalid frozen Worker result")
            if store:
                store.save_record("evaluation-results", key, exp.to_dict())
        else:
            exp = Experience.from_dict(saved)
        results.append(exp)
    return results


def transfer(sources: Mapping[str, Harness], targets: Mapping[str, Sequence[Task]],
             worker_factory, *, store: SnapshotStore | None = None) -> dict:
    """Fresh target Worker per source/target pair; transfer harness artifacts only.

    worker_factory(target_name) must create an isolated target environment with
    its original tools and permissions. No target history enters adaptation.
    """
    return {source: {target: evaluate_frozen(worker_factory(target), tasks, harness,
                         store=store, evaluation_id=f"transfer:{source}:{target}")
                     for target, tasks in targets.items()}
            for source, harness in sources.items()}
