"""Guards against accidentally feeding debug/smoke-test runs into paper-facing summaries,
tables, and figures. Only results from a run directory prefixed `full_run_` AND logging the
full configured number of rounds may be aggregated -- everything else (local_debug_*,
test_run_*, partial/interrupted runs) is refused loudly rather than silently included.
"""

from __future__ import annotations

import csv
import os

ALLOWED_PREFIX = "full_run"
DEBUG_PREFIXES = ("test_run_", "local_debug_")


class DebugRunError(AssertionError):
    """Raised when a debug/smoke-test or incomplete run would otherwise leak into a paper
    output (summary CSV, table, or figure)."""


def _run_folder_name(csv_path: str) -> str:
    return os.path.basename(os.path.dirname(csv_path))


def _count_rounds(csv_path: str) -> int:
    with open(csv_path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def assert_full_run(csv_path: str, expected_rounds: int) -> None:
    """Raises DebugRunError if `csv_path` is not a legitimate, complete full_run_* result.

    Checks (in order, so the message names the actual problem):
      1. folder name does not match a known debug prefix (test_run_*, local_debug_*)
      2. folder name starts with the required `full_run_` prefix
      3. the run logged >= expected_rounds rows (a truncated/interrupted run is not a "full" run)
    """
    if not os.path.exists(csv_path):
        raise DebugRunError(f"Cannot validate a run that does not exist on disk: {csv_path}")

    folder = _run_folder_name(csv_path)

    for bad_prefix in DEBUG_PREFIXES:
        if folder.startswith(bad_prefix):
            raise DebugRunError(
                f"Refusing to aggregate a debug/smoke-test run into a paper output: "
                f"'{folder}' matches the debug prefix '{bad_prefix}' ({csv_path}). "
                f"Only {ALLOWED_PREFIX}_* results may feed summaries, T1-T4, and F1-F7."
            )

    if not folder.startswith(ALLOWED_PREFIX + "_"):
        raise DebugRunError(
            f"Refusing to aggregate a non-full-run result into a paper output: "
            f"'{folder}' does not start with '{ALLOWED_PREFIX}_' ({csv_path}). "
            f"Only {ALLOWED_PREFIX}_* results may feed summaries, T1-T4, and F1-F7."
        )

    n_rounds = _count_rounds(csv_path)
    if n_rounds < expected_rounds:
        raise DebugRunError(
            f"Refusing to aggregate an incomplete run into a paper output: '{folder}' logged "
            f"{n_rounds} round(s), expected >= {expected_rounds} ({csv_path}). This looks like "
            f"a debug/smoke-test or interrupted run, not a completed full_run result."
        )


def assert_all_full_run(csv_paths: list[str], expected_rounds: int) -> None:
    """Convenience wrapper: validates every path, raising on the first failure."""
    for path in csv_paths:
        assert_full_run(path, expected_rounds)
