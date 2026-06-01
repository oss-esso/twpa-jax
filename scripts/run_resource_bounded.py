"""Run one command with conservative time, memory, and artifact-size limits."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _process_tree_pids(root_pid: int) -> set[int]:
    """Return the live root process and descendants without spawning helpers."""
    parent_by_pid: dict[int, int] = {}
    if os.name == "nt":
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return {root_pid}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            ok = ctypes.windll.kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                parent_by_pid[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                ok = ctypes.windll.kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            ctypes.windll.kernel32.CloseHandle(snapshot)
    else:
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text(encoding="utf-8").split()
                parent_by_pid[int(fields[0])] = int(fields[3])
            except (OSError, ValueError, IndexError):
                continue

    tree = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parent_by_pid.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def _working_set_bytes(pid: int) -> int | None:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not process:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(process)

    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _cpu_seconds(pid: int) -> float | None:
    if os.name == "nt":
        class FILETIME(ctypes.Structure):
            _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

        process = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
        if not process:
            return None
        try:
            creation = FILETIME()
            exit_time = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            ok = ctypes.windll.kernel32.GetProcessTimes(
                process,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            if not ok:
                return None
            ticks = (
                (kernel.high << 32)
                + kernel.low
                + (user.high << 32)
                + user.low
            )
            return ticks / 10_000_000.0
        finally:
            ctypes.windll.kernel32.CloseHandle(process)

    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        fields = stat.read_text(encoding="utf-8").split()
        return (float(fields[13]) + float(fields[14])) / os.sysconf("SC_CLK_TCK")
    return None


def _directory_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _tree_metrics(root_pid: int) -> tuple[int | None, float | None, int]:
    pids = _process_tree_pids(root_pid)
    memory_values = [_working_set_bytes(pid) for pid in pids]
    cpu_values = [_cpu_seconds(pid) for pid in pids]
    memory = [value for value in memory_values if value is not None]
    cpu = [value for value in cpu_values if value is not None]
    return (
        sum(memory) if memory else None,
        sum(cpu) if cpu else None,
        len(pids),
    )


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Terminate descendants first. taskkill /T can block behind a waiting
        # Python parent, which defeats enforcement for orchestration scripts.
        for pid in sorted(_process_tree_pids(process.pid), reverse=True):
            handle = ctypes.windll.kernel32.OpenProcess(0x0001, False, pid)
            if not handle:
                continue
            try:
                ctypes.windll.kernel32.TerminateProcess(handle, 1)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
    else:
        os.killpg(process.pid, signal.SIGTERM)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-memory-mib", type=float, default=1024.0)
    parser.add_argument("--max-disk-mib", type=float, default=256.0)
    parser.add_argument("--disk-root", type=Path, default=None)
    parser.add_argument("--log-json", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Provide a command after --")

    start = time.perf_counter()
    peak_memory = 0
    peak_disk = _directory_bytes(args.disk_root)
    max_cpu_seconds = 0.0
    peak_process_count = 0
    reason = "completed"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(command, creationflags=creationflags)
    try:
        while process.poll() is None:
            elapsed = time.perf_counter() - start
            memory, cpu_seconds, process_count = _tree_metrics(process.pid)
            disk = _directory_bytes(args.disk_root)
            peak_memory = max(peak_memory, memory or 0)
            peak_disk = max(peak_disk, disk)
            max_cpu_seconds = max(max_cpu_seconds, cpu_seconds or 0.0)
            peak_process_count = max(peak_process_count, process_count)
            if elapsed > args.timeout_s:
                reason = "timeout"
                _terminate(process)
                break
            if memory is not None and memory > args.max_memory_mib * 1024**2:
                reason = "memory_limit"
                _terminate(process)
                break
            if disk > args.max_disk_mib * 1024**2:
                reason = "disk_limit"
                _terminate(process)
                break
            time.sleep(0.2)
    finally:
        returncode = process.wait()

    elapsed = time.perf_counter() - start
    report = {
        "command": command,
        "returncode": returncode,
        "reason": reason,
        "elapsed_s": elapsed,
        "cpu_seconds": max_cpu_seconds,
        "average_cpu_percent": 100.0 * max_cpu_seconds / max(elapsed, 1e-12),
        "peak_working_set_mib": peak_memory / 1024**2,
        "peak_disk_mib": peak_disk / 1024**2,
        "peak_process_count": peak_process_count,
        "metric_scope": "process_tree",
        "limits": {
            "timeout_s": args.timeout_s,
            "max_memory_mib": args.max_memory_mib,
            "max_disk_mib": args.max_disk_mib,
        },
    }
    args.log_json.parent.mkdir(parents=True, exist_ok=True)
    args.log_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if reason == "completed" and returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
