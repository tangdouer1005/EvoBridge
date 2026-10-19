"""The only evolving state: domain memory, operations, and execution policy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping


ROOT_DOCUMENTS = frozenset({"memory.md", "CONTEXT.md"})


def _name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid harness path")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or
            any(part in (".", "..") or part.startswith(".") for part in path.parts)):
        raise ValueError("Harness paths must be canonical relative paths")
    if value not in ROOT_DOCUMENTS and not (len(path.parts) >= 2 and path.parts[0] == "skills"):
        raise ValueError("Harness files must be memory.md, CONTEXT.md, or inside skills/")
    return value


def _text(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("Harness contents must be UTF-8 text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Harness contents must be UTF-8 text") from exc
    return value


def _plain_path(path: Path) -> Path:
    """Reject links in the supplied path, including existing parent components."""
    path = path.absolute()
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            raise ValueError("Symbolic links are not allowed")
    return path


def _regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Expected an ordinary, unlinked file")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_text(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class Harness:
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.files, Mapping):
            raise ValueError("Harness files must be a mapping")
        values = {_name(name): _text(text) for name, text in self.files.items()}
        if not ROOT_DOCUMENTS.issubset(values):
            raise ValueError("Harness requires memory.md and CONTEXT.md")
        for name in values:
            if any(str(parent) in values for parent in PurePosixPath(name).parents):
                raise ValueError("A harness file cannot also be a directory")
        object.__setattr__(self, "files", MappingProxyType(dict(sorted(values.items()))))

    @property
    def id(self) -> str:
        return hashlib.sha256(_canonical(dict(self.files))).hexdigest()

    @classmethod
    def from_directory(cls, path: str | Path) -> Harness:
        root = _plain_path(Path(path))
        if not root.is_dir():
            raise ValueError("Harness directory does not exist")
        files = {}
        for directory, directories, names in os.walk(root, followlinks=False):
            for name in directories:
                entry = Path(directory) / name
                _plain_path(entry)
                relative = entry.relative_to(root).as_posix()
                if relative != "skills":
                    _name(relative)
            for name in names:
                entry = Path(directory) / name
                _plain_path(entry)
                _regular(entry)
                relative = _name(entry.relative_to(root).as_posix())
                try:
                    files[relative] = entry.read_bytes().decode("utf-8")
                except UnicodeError as exc:
                    raise ValueError("Harness files must be UTF-8 text") from exc
        return cls(files)

    def write_to(self, path: str | Path) -> None:
        """Materialize into a fresh directory or overwrite the same file set.

        Unexpected files are rejected rather than deleted. This prevents stale
        operations from silently surviving an export of a smaller harness.
        """
        root = _plain_path(Path(path))
        if root.exists() and not root.is_dir():
            raise ValueError("Harness destination must be a directory")
        if root.exists():
            for directory, directories, names in os.walk(root, followlinks=False):
                for name in directories:
                    entry = _plain_path(Path(directory) / name)
                    relative = entry.relative_to(root).as_posix()
                    if relative != "skills":
                        _name(relative)
                for name in names:
                    entry = _plain_path(Path(directory) / name)
                    _regular(entry)
                    if entry.relative_to(root).as_posix() not in self.files:
                        raise ValueError("Destination contains files outside this harness")
        # Preflight every destination before changing any existing file.
        for name in self.files:
            destination = _plain_path(root / name)
            if destination.exists():
                _regular(destination)
            for parent in destination.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError("A destination parent is not a directory")
                if parent == root:
                    break
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name, content in self.files.items():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _atomic_text(destination, content)


class Candidate:
    """A private working copy; mutations never alter the published Harness."""

    def __init__(self, harness: Harness):
        if not isinstance(harness, Harness):
            raise TypeError("Candidate requires a Harness")
        self._base = harness
        self._files = dict(harness.files)

    @property
    def base_id(self) -> str:
        return self._base.id

    def read(self, path: str) -> str:
        return self._files[_name(path)]

    def write(self, path: str, content: str) -> None:
        name, text = _name(path), _text(content)
        changed = {**self._files, name: text}
        Harness(changed)  # Validate file/directory collisions before mutation.
        self._files[name] = text

    def delete(self, path: str) -> None:
        name = _name(path)
        if name in ROOT_DOCUMENTS:
            raise ValueError("The two root harness documents cannot be deleted")
        del self._files[name]

    def list_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._files))

    def freeze(self) -> Harness:
        return Harness(self._files)

    def copy(self) -> Candidate:
        """Detach a working copy while preserving its base and file diff."""
        copy = Candidate(self._base)
        copy._files = dict(self._files)
        return copy

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(name for name in sorted(self._base.files.keys() | self._files.keys())
                     if self._base.files.get(name) != self._files.get(name))
