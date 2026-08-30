"""Run every lesson's ``python -m ... --help`` entrypoint in isolation.

This is an explicit, runtime-level maintenance gate.  It is intentionally not
part of ``make verify`` because importing all lesson entrypoints takes several
minutes even when checks run concurrently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

try:
    from scripts.lesson_contracts import discover_lesson_contracts
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from lesson_contracts import discover_lesson_contracts

HELP_PATTERN = re.compile(r"^usage:", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class EntrypointHelpResult:
    module: str
    status: str
    returncode: int | None
    elapsed_seconds: float
    error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class EntrypointHelpAudit:
    results: tuple[EntrypointHelpResult, ...]
    workers: int
    timeout_seconds: float
    elapsed_seconds: float

    @property
    def summary(self) -> dict[str, int | float | bool | str]:
        passed = sum(result.passed for result in self.results)
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "workers": self.workers,
            "timeout_seconds": self.timeout_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "temporary_roots_removed": True,
            "network_policy": "offline environment flags and isolated caches; sockets not blocked",
            "ok": passed == len(self.results),
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail(value: str | bytes | None, limit: int = 2_000) -> str:
    return _coerce_text(value)[-limit:]


def _entrypoint_environment(root: Path, case_root: Path) -> dict[str, str]:
    """Build a deterministic CPU/offline environment rooted outside the repo."""

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(root)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DIFFUSERS_OFFLINE": "1",
            "DLHUB_OUTPUTS_DIR": str(case_root / "outputs"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(case_root / "hf"),
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(case_root / "mpl"),
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": pythonpath,
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_HOME": str(case_root / "torch"),
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "XDG_CACHE_HOME": str(case_root / "cache"),
        }
    )
    return environment


def _run_help(
    module: str,
    *,
    root: Path,
    temporary_root: Path,
    timeout_seconds: float,
) -> EntrypointHelpResult:
    case_root = temporary_root / module.replace(".", "_")
    case_root.mkdir(parents=True)
    environment = _entrypoint_environment(root, case_root)
    command = [sys.executable, "-m", module, "--help"]
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=case_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return EntrypointHelpResult(
            module=module,
            status="timeout",
            returncode=None,
            elapsed_seconds=time.monotonic() - started,
            error=f"exceeded {timeout_seconds:g}s timeout",
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
        )

    elapsed_seconds = time.monotonic() - started
    if process.returncode != 0:
        return EntrypointHelpResult(
            module=module,
            status="exit-error",
            returncode=process.returncode,
            elapsed_seconds=elapsed_seconds,
            error=f"help command exited with {process.returncode}",
            stdout_tail=_tail(process.stdout),
            stderr_tail=_tail(process.stderr),
        )

    help_output = f"{process.stdout}\n{process.stderr}"
    if not HELP_PATTERN.search(help_output):
        return EntrypointHelpResult(
            module=module,
            status="missing-help",
            returncode=process.returncode,
            elapsed_seconds=elapsed_seconds,
            error="exit 0 without an argparse-style usage line",
            stdout_tail=_tail(process.stdout),
            stderr_tail=_tail(process.stderr),
        )

    return EntrypointHelpResult(
        module=module,
        status="passed",
        returncode=process.returncode,
        elapsed_seconds=elapsed_seconds,
    )


def audit_lesson_entrypoints(
    *,
    root: Path | None = None,
    workers: int = 4,
    timeout_seconds: float = 30.0,
    progress_stream: TextIO | None = None,
) -> EntrypointHelpAudit:
    root = (root or repo_root()).resolve()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be at least 1")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")

    contracts = discover_lesson_contracts(root)
    if not contracts:
        raise ValueError(f"no lesson contracts discovered under {root / 'tracks'}")
    modules = [contract.entrypoint_module for contract in contracts]
    missing = [contract.key for contract in contracts if contract.entrypoint_module is None]
    if missing:
        raise ValueError(f"lesson contracts have missing entrypoints: {missing!r}")
    entrypoint_modules = [module for module in modules if module is not None]
    if len(entrypoint_modules) != len(set(entrypoint_modules)):
        raise ValueError("lesson contract inventory contains duplicate entrypoint modules")

    started = time.monotonic()
    results: list[EntrypointHelpResult] = []
    with tempfile.TemporaryDirectory(prefix="dlhub-entrypoint-help-") as temporary:
        temporary_root = Path(temporary).resolve()
        if temporary_root == root or root in temporary_root.parents:
            raise ValueError(f"temporary root must be outside the repository: {temporary_root}")
        if progress_stream is not None:
            print(f"lesson entrypoint help: temporary root {temporary_root}", file=progress_stream)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_help,
                    module,
                    root=root,
                    temporary_root=temporary_root,
                    timeout_seconds=timeout_seconds,
                ): module
                for module in entrypoint_modules
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                module = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive reporting boundary
                    result = EntrypointHelpResult(
                        module=module,
                        status="internal-error",
                        returncode=None,
                        elapsed_seconds=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results.append(result)
                if progress_stream is not None and (
                    completed % 50 == 0 or completed == len(futures)
                ):
                    print(
                        f"lesson entrypoint help: {completed}/{len(futures)} completed",
                        file=progress_stream,
                    )

    return EntrypointHelpAudit(
        results=tuple(sorted(results, key=lambda result: result.module)),
        workers=workers,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=time.monotonic() - started,
    )


def _format_summary(audit: EntrypointHelpAudit) -> str:
    summary = audit.summary
    lines = [
        f"lesson entrypoint help: {summary['passed']}/{summary['total']} passed",
        "- command: python -m <entrypoint> --help",
        f"- workers: {summary['workers']}, per-entrypoint timeout: {summary['timeout_seconds']:g}s",
        "- environment: CPU-only, offline flags, deterministic hash, compute threads=1",
        "- network boundary: isolated caches/offline flags; sockets are not blocked",
        "- cwd/caches/outputs: isolated temporary roots (removed)",
        f"- elapsed: {summary['elapsed_seconds']:.2f}s",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every lesson's module entrypoint --help command in offline isolation."
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero on any failed help run."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent subprocesses (default: 4)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-entrypoint timeout in seconds (default: 30).",
    )
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required because this opt-in gate takes several minutes")

    try:
        audit = audit_lesson_entrypoints(
            workers=args.workers,
            timeout_seconds=args.timeout,
            progress_stream=sys.stderr,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "summary": audit.summary,
                    "results": [asdict(result) for result in audit.results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_format_summary(audit))
        failures = [result for result in audit.results if not result.passed]
        if failures:
            print(f"lesson entrypoint help: FAILED ({len(failures)} errors)")
            for result in failures:
                print(f"- {result.module}: {result.status}: {result.error}")
                if result.stderr_tail:
                    print(f"  stderr tail: {result.stderr_tail!r}")
        else:
            print("lesson entrypoint help: OK")

    return 0 if audit.summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
