"""Launch one repository process with explicit output files."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    # Python's environment mapping has one canonical PATH entry.  Supplying a
    # copy explicitly avoids PowerShell's case-duplicate PATH issue while
    # retaining the complete interpreter/DLL search path.
    environment = dict(os.environ)
    environment["PATH"] = os.environ.get("PATH", "")
    with args.stdout.open("w", encoding="utf-8") as stdout, args.stderr.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        )
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
