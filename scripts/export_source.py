#!/usr/bin/env python3
"""Export tracked source files with normalized ZIP metadata and no Git history."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


EXCLUDED = {".git", ".github", ".venv", "__pycache__", "build", "dist", "runtime"}


def export_source(root: Path, destination: Path) -> int:
    root = root.resolve()
    destination = destination.resolve()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = []
    for name in sorted(filter(None, names)):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid tracked path")
        if any(part in EXCLUDED for part in relative.parts):
            continue
        source = root / name
        if source == destination or not source.exists():
            continue
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError("Source archives require ordinary files within the checkout")
        files.append((name, source.read_bytes()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in files:
            entry = zipfile.ZipInfo("EvoBridge/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/evobridge-source.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    count = export_source(root, Path(args.output))
    print(f"Exported {count} source files.")


if __name__ == "__main__":
    main()
