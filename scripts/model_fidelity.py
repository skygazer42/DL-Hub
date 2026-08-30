"""Inspect and validate the audited DL-Hub Model Zoo fidelity ledger."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCOPE_NOTE = (
    "Unlisted artifacts are unreviewed; registration counts do not imply "
    "paper-faithful implementations."
)


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    return root


def _record_payload(record) -> dict[str, object]:
    payload = asdict(record)
    payload["level"] = record.level.value
    return payload


def _format_summary(summary: dict[str, int]) -> str:
    return "\n".join(
        [
            f"zoo fidelity: {summary['audited_groups']} audited groups / "
            f"{summary['audited_artifacts']} source artifacts",
            f"- reference: {summary['reference']} groups",
            f"- compact: {summary['compact']} groups",
            f"- baseline-alias: {summary['baseline-alias']} groups",
            f"- {SCOPE_NOTE}",
        ]
    )


def _format_audit_pressure(pressure: dict[str, int | float]) -> str:
    return (
        "audit pressure: "
        f"{pressure['total_registration_ids']} registration IDs / "
        f"{pressure['audited_artifacts']} audited source artifacts = "
        f"{pressure['registrations_per_audited_artifact']:.2f} "
        f"(maximum {pressure['max_registrations_per_audited_artifact']:.2f})"
    )


def _format_baseline_summary(inventory: dict[str, object]) -> str:
    summary = inventory["summary"]
    assert isinstance(summary, dict)
    return (
        "baseline wrappers: "
        f"{summary['total_wrappers']} total / "
        f"{summary['audited_wrappers']} audited / "
        f"{summary['source_inferred_alias_wrappers']} source-inferred aliases / "
        f"{summary['unreviewed_wrappers']} unidentified "
        f"(debt reduction {summary['debt_reduction']})"
    )


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    from dlhub.zoo_fidelity import (
        BASELINE_INVENTORY_PATH,
        FidelityLevel,
        build_baseline_inventory,
        get_fidelity_record,
        iter_fidelity_records,
        summarize_audit_pressure,
        summarize_fidelity,
        validate_audit_pressure,
        validate_baseline_inventory,
        validate_fidelity_records,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate records, source paths, and the registration-to-audit growth budget.",
    )
    parser.add_argument("--json", action="store_true", help="Print the selected inventory as JSON.")
    parser.add_argument("--list", action="store_true", help="List selected audit groups.")
    parser.add_argument("--show", metavar="KEY", help="Show one audit group by stable key.")
    parser.add_argument(
        "--write-baseline-inventory",
        action="store_true",
        help="Regenerate the deterministic direct baseline-wrapper inventory.",
    )
    parser.add_argument(
        "--level",
        choices=[level.value for level in FidelityLevel if level is not FidelityLevel.UNREVIEWED],
        help="Filter --list/--json output by reviewed fidelity level.",
    )
    args = parser.parse_args(argv)

    try:
        records = (
            (get_fidelity_record(args.show),) if args.show else iter_fidelity_records(args.level)
        )
    except KeyError as exc:
        parser.error(str(exc))

    summary = summarize_fidelity(records)
    baseline_inventory = build_baseline_inventory(root)
    if args.write_baseline_inventory:
        from dlhub._atomic import atomic_write

        encoded = (
            json.dumps(baseline_inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        def _write_inventory(handle) -> None:
            handle.write(encoded)

        inventory_path = atomic_write(root / BASELINE_INVENTORY_PATH, _write_inventory)
        print(f"wrote baseline inventory: {inventory_path.relative_to(root).as_posix()}")

    errors = validate_fidelity_records(root) if args.check else []
    baseline_inventory_errors = validate_baseline_inventory(root) if args.check else []
    errors.extend(baseline_inventory_errors)
    audit_pressure = None
    if args.check:
        from dlhub.project_stats import compute_stats

        total_registration_ids = compute_stats(root).total_zoo_ids
        audit_pressure = summarize_audit_pressure(total_registration_ids)
        errors.extend(validate_audit_pressure(total_registration_ids))

    if args.json:
        print(
            json.dumps(
                {
                    "scope": "audited-groups-only",
                    "scope_note": SCOPE_NOTE,
                    "summary": summary,
                    "audit_pressure": audit_pressure,
                    "baseline_inventory": baseline_inventory,
                    "baseline_inventory_errors": baseline_inventory_errors,
                    "records": [_record_payload(record) for record in records],
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_format_summary(summary))
        print(_format_baseline_summary(baseline_inventory))
        if audit_pressure is not None:
            print(_format_audit_pressure(audit_pressure))
        if args.list or args.show or args.level:
            for record in records:
                print(f"{record.key}\t{record.level.value}\t{len(record.artifacts)} artifacts")
                if args.show:
                    print(f"  {record.summary}")
                    for mechanism in record.missing_mechanisms:
                        print(f"  missing: {mechanism}")
                    print(f"  next: {record.next_action}")

        if args.check:
            if errors:
                print(f"zoo fidelity: FAILED ({len(errors)} errors)")
                for error in errors:
                    print(f"- {error}")
            else:
                print("zoo fidelity: OK")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
