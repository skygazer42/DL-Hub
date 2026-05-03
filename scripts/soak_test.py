from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "outputs" / "soak_test" / "latest.jsonl"


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]


CHECKS: tuple[Check, ...] = (
    Check("pytest", (sys.executable, "-m", "pytest")),
    Check(
        "topic_audit",
        (
            sys.executable,
            "-c",
            "from dlhub.topic_coverage import coverage_report, validate_topic_coverage; "
            "validate_topic_coverage(); "
            "r=coverage_report(); "
            "print(f'topic_audit requested={r.requested_count} covered={r.covered_count} missing={len(r.missing_topics)}')",
        ),
    ),
    Check(
        "ruff_changed_surfaces",
        (
            sys.executable,
            "-m",
            "ruff",
            "check",
            "Llms/__init__.py",
            "dlhub/topic_coverage.py",
            "dlhub/research_streams.py",
            "dlhub/framework_adapters.py",
            "dlhub/method_kits.py",
            "dlhub/vision/denoising/derainformer.py",
            "dlhub/vision/denoising/did_mdn.py",
            "dlhub/vision/denoising/rcdnet.py",
            "dlhub/vision/denoising/transweather.py",
            "tests/test_topic_coverage.py",
            "tests/test_zoo_conventions_smoke.py",
        ),
    ),
    Check(
        "compile_changed_surfaces",
        (
            sys.executable,
            "-m",
            "py_compile",
            "Llms/__init__.py",
            "dlhub/topic_coverage.py",
            "dlhub/research_streams.py",
            "dlhub/framework_adapters.py",
            "dlhub/method_kits.py",
            "dlhub/vision/denoising/derainformer.py",
            "dlhub/vision/denoising/did_mdn.py",
            "dlhub/vision/denoising/rcdnet.py",
            "dlhub/vision/denoising/transweather.py",
            "tests/test_topic_coverage.py",
            "tests/test_zoo_conventions_smoke.py",
        ),
    ),
    Check("diff_check", ("git", "diff", "--check")),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_event(log_path: Path, event: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def run_check(check: Check, *, log_path: Path, round_index: int) -> bool:
    start = time.monotonic()
    started_at = _utc_now()
    proc = subprocess.run(
        check.command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.monotonic() - start
    event = {
        "type": "check",
        "round": round_index,
        "name": check.name,
        "command": list(check.command),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    _write_event(log_path, event)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[round {round_index}] {status} {check.name} ({elapsed:.1f}s)", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
    return proc.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated repository verification checks.")
    parser.add_argument("--hours", type=float, default=5.0, help="Minimum soak duration in hours.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="JSONL log path.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Optional maximum number of rounds; 0 means run until duration elapses.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue the soak after a failed check; default stops on first failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    duration_seconds = max(0.0, float(args.hours) * 3600.0)
    started = time.monotonic()
    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()

    _write_event(
        log_path,
        {
            "type": "start",
            "started_at": _utc_now(),
            "duration_seconds": duration_seconds,
            "checks": [check.name for check in CHECKS],
        },
    )

    round_index = 0
    failures = 0
    while True:
        if args.max_rounds and round_index >= int(args.max_rounds):
            break
        if round_index > 0 and time.monotonic() - started >= duration_seconds:
            break

        round_index += 1
        print(f"[round {round_index}] starting", flush=True)
        for check in CHECKS:
            ok = run_check(check, log_path=log_path, round_index=round_index)
            if not ok:
                failures += 1
                if not args.continue_on_failure:
                    _write_event(
                        log_path,
                        {
                            "type": "finish",
                            "finished_at": _utc_now(),
                            "rounds": round_index,
                            "failures": failures,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        },
                    )
                    return 1

    elapsed = time.monotonic() - started
    _write_event(
        log_path,
        {
            "type": "finish",
            "finished_at": _utc_now(),
            "rounds": round_index,
            "failures": failures,
            "elapsed_seconds": round(elapsed, 3),
        },
    )
    print(f"soak complete: rounds={round_index} failures={failures} elapsed={elapsed:.1f}s")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
