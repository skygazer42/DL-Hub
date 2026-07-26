"""Shared helpers for the scripts/run_lesson.py test suite.

Not prefixed with test_ so pytest does not collect it.
"""

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
