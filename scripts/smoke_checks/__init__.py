"""Smoke checks for DL-Hub, organized by track/topic.

Each module exposes small check functions (or a ``run()``) whose bodies were
moved verbatim from the original monolithic ``scripts/smoke_check.py`` main().
The entry point ``scripts/smoke_check.py`` wires them together and keeps the
torch-availability handling.
"""
