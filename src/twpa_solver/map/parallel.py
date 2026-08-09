"""Small process-isolated job scheduling primitive for map columns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from typing import TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")


def run_isolated_jobs(
    jobs: Iterable[Job], worker: Callable[[Job], Result], max_workers: int,
) -> list[Result]:
    """Run independent subprocess-launching jobs with bounded parent threads."""
    pending = list(jobs)
    if not pending:
        return []
    workers = max(1, min(int(max_workers), len(pending)))
    if workers == 1:
        return [worker(job) for job in pending]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, job) for job in pending]
        return [future.result() for future in as_completed(futures)]
