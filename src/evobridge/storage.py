"""Private snapshots, compare-and-swap publication, and immutable local records."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator

from .harness import Candidate, Harness, _canonical, _plain_path, _regular


def _kind(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Invalid record kind")
    return value


def _key(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Record keys must be nonempty strings")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid snapshot identity")
    return value


def _object(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Records must be JSON objects")
    # Round-trip before writing: reject non-JSON values and isolate caller state.
    return json.loads(_canonical(value))


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = _plain_path(Path(root))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        for name in ("snapshots", "records"):
            self._directory(self.root / name)
        lock = self.root / ".lock"
        _plain_path(lock)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock, flags, 0o600)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")  # Windows locks a byte range.
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        _regular(lock)

    def _directory(self, path: Path) -> Path:
        _plain_path(path)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        return path

    @contextmanager
    def _locked(self) -> Iterator[None]:
        _plain_path(self.root / ".lock")
        _regular(self.root / ".lock")
        with (self.root / ".lock").open("r+b") as lock:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read(path: Path) -> dict:
        _plain_path(path)
        _regular(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Corrupt stored object")
        return value

    @staticmethod
    def _sync_directory(path: Path) -> None:
        if os.name != "nt":
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _write(self, path: Path, data: dict, *, exclusive: bool = False) -> None:
        _plain_path(path)
        if path.exists():
            _regular(path)
        payload = _canonical(data) + b"\n"
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if exclusive:
                # link is exclusive and exposes only a completely written file.
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
            if os.path.exists(temporary):
                os.unlink(temporary)
            self._sync_directory(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _snapshot(self, identity: str) -> Harness:
        identity = _identity(identity)
        value = self._read(self.root / "snapshots" / f"{identity}.json")
        harness = Harness(value["files"])
        if harness.id != identity:
            raise ValueError("Snapshot integrity failure")
        return harness

    def _save_snapshot(self, harness: Harness) -> None:
        path = self.root / "snapshots" / f"{harness.id}.json"
        if path.exists():
            if self._snapshot(harness.id).files != harness.files:
                raise ValueError("Snapshot integrity failure")
        else:
            self._write(path, {"files": dict(harness.files)}, exclusive=True)

    def _current(self) -> Harness:
        pointer = self.root / "current.json"
        if not pointer.exists():
            raise ValueError("Snapshot store is not initialized")
        return self._snapshot(self._read(pointer)["snapshot_id"])

    def initialize(self, harness: Harness, config_identity: str) -> Harness:
        if not isinstance(harness, Harness):
            raise TypeError("initialize requires a Harness")
        if not isinstance(config_identity, str) or not config_identity:
            raise ValueError("A frozen configuration identity is required")
        with self._locked():
            config = self.root / "config.json"
            expected_config = {"identity": config_identity, "initial_id": harness.id}
            if config.exists():
                if self._read(config) != expected_config:
                    raise ValueError("Frozen configuration or initial harness changed")
            else:
                self._write(config, expected_config, exclusive=True)
            if (self.root / "current.json").exists():
                return self._current()
            self._save_snapshot(harness)
            self._write(self.root / "current.json", {"snapshot_id": harness.id}, exclusive=True)
            return harness

    def current(self) -> Harness:
        with self._locked():
            return self._current()

    def publish(self, candidate: Candidate, expected_id: str) -> Harness:
        if not isinstance(candidate, Candidate):
            raise TypeError("publish requires a Candidate")
        _identity(expected_id)
        harness = candidate.freeze()
        if candidate.base_id != expected_id:
            raise ValueError("Candidate was created from a different base")
        with self._locked():
            if self._current().id != expected_id:
                raise ValueError("Stale harness: another update was published")
            self._save_snapshot(harness)
            self._write(self.root / "current.json", {"snapshot_id": harness.id})
        return harness

    def save_event(self, kind: str, data: dict) -> None:
        event = {"kind": _kind(kind), "data": _object(data)}
        with self._locked():
            path = _plain_path(self.root / "events.jsonl")
            if path.exists():
                _regular(path)
            flags = os.O_APPEND | os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
            with os.fdopen(fd, "ab") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(_canonical(event) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _record_path(self, kind: str, key: str) -> Path:
        return self.root / "records" / _kind(kind) / f"{_key(key)}.json"

    def _read_record(self, kind: str, key: str) -> dict | None:
        path = self._record_path(kind, key)
        _plain_path(path)
        if not path.exists():
            return None
        record = self._read(path)
        if record.get("key") != key or not isinstance(record.get("data"), dict):
            raise ValueError("Record integrity failure")
        return record

    def save_record(self, kind: str, key: str, data: dict) -> bool:
        """Return True only for the process that creates this immutable record.

        False denotes an identical existing record. Callers can use the result
        to claim a model/execution start without a check-then-write race.
        """
        value = _object(data)
        path = self._record_path(kind, key)
        with self._locked():
            old = self._read_record(kind, key)
            if old is not None:
                if _canonical(old["data"]) != _canonical(value):
                    raise ValueError("An immutable record cannot be changed")
                return False
            directory = self._directory(path.parent)
            records = [self._read(p) for p in directory.glob("*.json")]
            sequence = max((r["sequence"] for r in records), default=0) + 1
            self._write(path, {"key": key, "sequence": sequence, "data": value}, exclusive=True)
            return True

    def read_record(self, kind: str, key: str) -> dict | None:
        with self._locked():
            record = self._read_record(kind, key)
            return None if record is None else record["data"]

    def history(self, kind: str) -> list[dict]:
        directory = self.root / "records" / _kind(kind)
        with self._locked():
            _plain_path(directory)
            if not directory.exists():
                return []
            records = [self._read(p) for p in directory.glob("*.json")]
            records.sort(key=lambda r: r["sequence"])
            if len({r["sequence"] for r in records}) != len(records):
                raise ValueError("Record ordering integrity failure")
            return [record["data"] for record in records]
