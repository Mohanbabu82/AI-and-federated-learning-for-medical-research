"""Checkpoint/resume + progress logging for experiments.runner, so a long full_run.yaml sweep
survives interruptions on this 8 GB laptop: completed (method/ablation-config, seed) runs are
recorded to disk immediately after each finishes, and --resume skips anything already done.
"""

from __future__ import annotations

import datetime
import gc
import json
import os
import time


def checkpoint_path(results_dir: str) -> str:
    return os.path.join(results_dir, "_runner_checkpoint.json")


def progress_log_path(results_dir: str) -> str:
    return os.path.join(results_dir, "runner_progress.log")


def load_checkpoint(results_dir: str) -> dict:
    """Returns {"completed": {run_key: csv_path}}. Empty if no checkpoint exists yet."""
    path = checkpoint_path(results_dir)
    if not os.path.exists(path):
        return {"completed": {}}
    with open(path) as f:
        return json.load(f)


def mark_complete(results_dir: str, run_key: str, csv_path: str) -> None:
    """Atomically records one completed run so a crash mid-sweep can't lose already-finished
    work. Written after EVERY run, not batched."""
    os.makedirs(results_dir, exist_ok=True)
    state = load_checkpoint(results_dir)
    state["completed"][run_key] = csv_path

    tmp_path = checkpoint_path(results_dir) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, checkpoint_path(results_dir))  # atomic on both POSIX and Windows


def is_complete(results_dir: str, run_key: str) -> bool:
    state = load_checkpoint(results_dir)
    csv_path = state["completed"].get(run_key)
    return csv_path is not None and os.path.exists(csv_path)


def log_progress(results_dir: str, message: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log_path(results_dir), "a") as f:
        f.write(f"[{timestamp}] {message}\n")


class ProgressTracker:
    """Tracks elapsed/remaining time across a sweep of `total` runs, logging progress+ETA to
    <results_dir>/runner_progress.log after every completed run, and freeing memory between
    runs (gc.collect()) since sequential CPU training in one long-lived process otherwise
    accumulates dataset/model objects."""

    def __init__(self, results_dir: str, total_runs: int, already_done: int = 0):
        self.results_dir = results_dir
        self.total_runs = total_runs
        self.done = already_done
        self.start_time = time.time()
        self._run_times: list[float] = []
        log_progress(results_dir, f"=== Sweep started: {total_runs} total runs, "
                                   f"{already_done} already complete (resumed) ===")

    def record_run(self, run_key: str, elapsed_sec: float, skipped: bool = False) -> None:
        if skipped:
            log_progress(self.results_dir, f"SKIP  (already complete) {run_key}")
            return

        self.done += 1
        self._run_times.append(elapsed_sec)
        avg = sum(self._run_times) / len(self._run_times)
        remaining = self.total_runs - self.done
        eta_sec = avg * remaining

        log_progress(
            self.results_dir,
            f"DONE  {run_key}  elapsed={elapsed_sec:.1f}s  "
            f"progress={self.done}/{self.total_runs} "
            f"({100 * self.done / self.total_runs:.1f}%)  "
            f"avg={avg:.1f}s/run  ETA={_fmt_hms(eta_sec)}",
        )

        # free memory between runs: sequential CPU training in one process otherwise
        # accumulates dataset/model objects across dozens-to-hundreds of runs
        gc.collect()


def _fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"
