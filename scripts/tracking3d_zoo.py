from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _summarize(obj) -> str:
    from dlhub.cli_utils import summarize_output

    return summarize_output(obj)


def _print_lines(lines: Iterable[str], *, limit: int = 80) -> None:
    from dlhub.cli_utils import print_limited

    print_limited(lines, limit=limit, annotate_fidelity=True)


def _save_leaderboard(
    *,
    path: Path,
    profile: str,
    variant: str,
    top_k: int,
    rank_by: str,
    run_results: list[dict[str, object]],
    leaderboard: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = {
            "profile": str(profile),
            "variant": str(variant),
            "top_k": int(top_k),
            "rank_by": str(rank_by),
            "run_results": run_results,
            "leaderboard": leaderboard,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        return

    if suffix == ".csv":
        rank_by_run: dict[str, int] = {
            str(row["run_name"]): int(row["rank"])
            for row in leaderboard
            if "run_name" in row and "rank" in row
        }
        fieldnames = [
            "rank",
            "arch_id",
            "family",
            "group",
            "year",
            "score",
            "run_name",
            "source",
            "ok",
            "returncode",
            "elapsed_sec",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in run_results:
                run_name = str(row.get("run_name", ""))
                out = {k: row.get(k, "") for k in fieldnames}
                out["rank"] = rank_by_run.get(run_name, "")
                writer.writerow(out)
        return

    raise ValueError(f"Unsupported leaderboard file extension: {path.suffix!r}. Use .json or .csv")


def _rank_successful_runs(
    ok_runs: list[dict[str, object]], *, rank_by: str
) -> list[dict[str, object]]:
    mode = str(rank_by).strip().lower()
    rows = [dict(x) for x in ok_runs]

    def _num(v: object, *, default: float) -> float:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, int | float):
            return float(v)
        return float(default)

    if mode == "recommend":
        rows.sort(
            key=lambda x: (
                -_num(x.get("score"), default=-1e9),
                _num(x.get("elapsed_sec"), default=1e9),
                str(x.get("arch_id", "")),
            )
        )
        return rows

    if mode == "elapsed":
        rows.sort(
            key=lambda x: (
                _num(x.get("elapsed_sec"), default=1e9),
                -_num(x.get("score"), default=-1e9),
                str(x.get("arch_id", "")),
            )
        )
        return rows

    raise ValueError(f"Unsupported rank mode: {rank_by!r}. Use one of: recommend, elapsed")


def _resolve_artifacts_dir(
    *,
    repo_root: Path,
    raw: str,
    profile: str,
    variant: str,
    top_k: int,
    now: datetime | None = None,
) -> Path:
    value = str(raw).strip()
    if not value:
        raise ValueError("save-artifacts-dir cannot be empty")
    if value.lower() != "auto":
        out = Path(value)
        if not out.is_absolute():
            out = repo_root / out
        return out

    t = now or datetime.now()
    stamp = t.strftime("%Y%m%d_%H%M%S")
    name = f"{profile}_{variant}_top{int(top_k)}_{stamp}"
    return repo_root / "outputs" / "pointcloud" / "tracking3d_artifacts" / name


def _write_markdown_report(
    *,
    path: Path,
    profile: str,
    variant: str,
    top_k: int,
    rank_by: str,
    run_results: list[dict[str, object]],
    leaderboard: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# Tracking3D Batch Report")
    lines.append("")
    lines.append(f"- profile: `{profile}`")
    lines.append(f"- variant: `{variant}`")
    lines.append(f"- top_k: `{int(top_k)}`")
    lines.append(f"- rank_by: `{str(rank_by)}`")
    lines.append(f"- runs: `{len(run_results)}`")
    lines.append(f"- success: `{sum(1 for x in run_results if bool(x.get('ok')))}`")
    lines.append("")
    lines.append("## Run Results")
    lines.append("")
    lines.append("| idx | arch_id | run_name | source | ok | recommend_score | elapsed_sec |")
    lines.append("|---:|---|---|---|:---:|---:|---:|")
    for row in run_results:
        lines.append(
            "| {idx} | {arch} | {run} | {source} | {ok} | {score} | {elapsed} |".format(
                idx=row.get("idx", ""),
                arch=row.get("arch_id", ""),
                run=row.get("run_name", ""),
                source=row.get("source", ""),
                ok="Y" if bool(row.get("ok")) else "N",
                score=row.get("score", ""),
                elapsed=row.get("elapsed_sec", ""),
            )
        )
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    if leaderboard:
        lines.append("| rank | arch_id | run_name | source | recommend_score | elapsed_sec |")
        lines.append("|---:|---|---|---|---:|---:|")
        for row in leaderboard:
            lines.append(
                "| {rank} | {arch} | {run} | {source} | {score} | {elapsed} |".format(
                    rank=row.get("rank", ""),
                    arch=row.get("arch_id", ""),
                    run=row.get("run_name", ""),
                    source=row.get("source", ""),
                    score=row.get("score", ""),
                    elapsed=row.get("elapsed_sec", ""),
                )
            )
    else:
        lines.append("_No successful runs._")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _save_artifacts(
    *,
    artifacts_dir: Path,
    profile: str,
    variant: str,
    top_k: int,
    rank_by: str,
    planned: list[tuple[int, object, str, list[str]]],
    run_results: list[dict[str, object]],
    leaderboard: list[dict[str, object]],
    run_logs: dict[str, dict[str, str]],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    commands = ["python " + " ".join(smoke_cmd[1:]) for _, _, _, smoke_cmd in planned]
    with (artifacts_dir / "commands.txt").open("w", encoding="utf-8") as f:
        for cmd in commands:
            f.write(cmd)
            f.write("\n")

    recommendations: list[dict[str, object]] = []
    for idx, rec, run_name, _ in planned:
        recommendations.append(
            {
                "idx": int(idx),
                "arch_id": str(getattr(rec, "arch_id", "")),
                "family": str(getattr(rec, "family", "")),
                "group": str(getattr(rec, "group", "")),
                "year": (
                    int(getattr(rec, "year")) if getattr(rec, "year", None) is not None else None
                ),
                "score": float(getattr(rec, "score", 0.0)),
                "reason": str(getattr(rec, "reason", "")),
                "run_name": run_name,
            }
        )

    payload = {
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": str(profile),
        "variant": str(variant),
        "top_k": int(top_k),
        "rank_by": str(rank_by),
        "num_planned": len(planned),
        "num_runs": len(run_results),
        "num_success": sum(1 for x in run_results if bool(x.get("ok"))),
        "recommendations": recommendations,
    }
    with (artifacts_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    with (artifacts_dir / "run_results.json").open("w", encoding="utf-8") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    with (artifacts_dir / "leaderboard_rows.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    _save_leaderboard(
        path=artifacts_dir / "leaderboard.json",
        profile=profile,
        variant=variant,
        top_k=top_k,
        rank_by=rank_by,
        run_results=run_results,
        leaderboard=leaderboard,
    )
    _save_leaderboard(
        path=artifacts_dir / "leaderboard.csv",
        profile=profile,
        variant=variant,
        top_k=top_k,
        rank_by=rank_by,
        run_results=run_results,
        leaderboard=leaderboard,
    )
    _write_markdown_report(
        path=artifacts_dir / "report.md",
        profile=profile,
        variant=variant,
        top_k=top_k,
        rank_by=rank_by,
        run_results=run_results,
        leaderboard=leaderboard,
    )

    for row in run_results:
        run_name = str(row.get("run_name", "run"))
        idx = int(row.get("idx", 0))
        stem = f"{idx:02d}_{run_name}"
        logs = run_logs.get(run_name, {})
        stdout_text = str(logs.get("stdout", ""))
        stderr_text = str(logs.get("stderr", ""))
        with (logs_dir / f"{stem}.stdout.log").open("w", encoding="utf-8") as f_out:
            f_out.write(stdout_text)
        with (logs_dir / f"{stem}.stderr.log").open("w", encoding="utf-8") as f_err:
            f_err.write(stderr_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracking3D local model zoo utilities (no downloads)."
    )
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument(
        "--timeline", action="store_true", help="Print a best-effort Tracking3D timeline."
    )
    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        metavar="PROFILE",
        help="Recommend Tracking3D architectures for a scenario profile.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available recommendation profiles and exit.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K recommendations to print with --recommend.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        help="Variant used with --recommend: tiny | small | base.",
    )
    parser.add_argument(
        "--emit-smoke-cmds",
        action="store_true",
        help="Print runnable smoke commands for each recommendation.",
    )
    parser.add_argument(
        "--run-smoke-cmds",
        action="store_true",
        help="Run emitted smoke commands and print a mini leaderboard.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Used with --run-smoke-cmds to reduce console output.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="recommend",
        choices=["recommend", "elapsed"],
        help="Ranking mode for successful smoke runs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one smoke command fails (with --run-smoke-cmds).",
    )
    parser.add_argument(
        "--save-leaderboard",
        type=str,
        default=None,
        help="Save run results/leaderboard to a file (.json or .csv). Requires --run-smoke-cmds.",
    )
    parser.add_argument(
        "--save-artifacts-dir",
        type=str,
        default=None,
        help=(
            "Save batch-run artifacts into a directory "
            "(commands, logs, run_results, leaderboard). Use 'auto' for timestamped path. "
            "Requires --run-smoke-cmds."
        ),
    )
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        metavar="ARCH_ID",
        help="Run a short sequence smoke on an arch id.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=4, help="Tracking sequence length.")
    parser.add_argument("--num-points", type=int, default=128, help="Number of points per frame.")
    parser.add_argument("--in-channels", type=int, default=3, help="Point feature channels.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of track classes.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    repo_root = Path(__file__).resolve().parents[1]
    from dlhub.pointcloud.tracking3d._recommend import list_profiles, recommend_arches
    from dlhub.pointcloud.tracking3d._timeline import entries
    from dlhub.pointcloud.tracking3d_zoo import build_local_model, list_local_arches

    args = parse_args()
    if args.emit_smoke_cmds and args.recommend is None:
        print("--emit-smoke-cmds is only valid with --recommend.")
        print("Tip: python scripts/tracking3d_zoo.py --recommend bev_priority --emit-smoke-cmds")
        return 2
    if args.run_smoke_cmds and args.recommend is None:
        print("--run-smoke-cmds is only valid with --recommend.")
        print("Tip: python scripts/tracking3d_zoo.py --recommend bev_priority --run-smoke-cmds")
        return 2
    if args.summary_only and not args.run_smoke_cmds:
        print("--summary-only is only valid with --run-smoke-cmds.")
        print(
            "Tip: python scripts/tracking3d_zoo.py --recommend bev_priority --run-smoke-cmds --summary-only"
        )
        return 2
    if args.save_leaderboard is not None and not args.run_smoke_cmds:
        print("--save-leaderboard is only valid with --run-smoke-cmds.")
        print(
            "Tip: python scripts/tracking3d_zoo.py --recommend bev_priority --run-smoke-cmds --save-leaderboard outputs/pointcloud/tracking3d_leaderboard.json"
        )
        return 2
    if args.save_artifacts_dir is not None and not args.run_smoke_cmds:
        print("--save-artifacts-dir is only valid with --run-smoke-cmds.")
        print(
            "Tip: python scripts/tracking3d_zoo.py --recommend bev_priority --run-smoke-cmds --save-artifacts-dir outputs/pointcloud/tracking3d_artifacts"
        )
        return 2
    if (
        not args.list
        and not args.timeline
        and args.recommend is None
        and not args.list_profiles
        and args.smoke is None
    ):
        print("Nothing to do. Try one of:")
        print("- python scripts/tracking3d_zoo.py --list")
        print("- python scripts/tracking3d_zoo.py --timeline")
        print("- python scripts/tracking3d_zoo.py --list-profiles")
        print("- python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 8")
        print(
            "- python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 5 --emit-smoke-cmds"
        )
        print(
            "- python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 3 --run-smoke-cmds"
        )
        print(
            "- python scripts/tracking3d_zoo.py --recommend bev_priority --top-k 3 --run-smoke-cmds --save-artifacts-dir outputs/pointcloud/tracking3d_artifacts"
        )
        print("- python scripts/tracking3d_zoo.py --smoke pctrk3d:ab3dmot_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Tracking3D local zoo")
        print(f"- total_arches={len(arches)}")
        print("")
        _print_lines(arches, limit=int(args.limit))

    if args.timeline:
        timeline = entries()
        if needle:
            timeline = [
                e
                for e in timeline
                if needle in e.family.lower()
                or needle in e.method.lower()
                or needle in e.group.lower()
            ]
        print("Tracking3D timeline")
        print(f"- total_families={len(timeline)}")
        print(f"- total_arches={len(arches)}")
        current_year = None
        for e in sorted(
            timeline, key=lambda x: (9999 if x.year is None else x.year, x.group, x.family)
        ):
            y = "unknown" if e.year is None else str(e.year)
            if y != current_year:
                print("")
                print(y)
                current_year = y
            print(f"- {e.family} [{e.group}]: {e.method} -> pctrk3d:{e.family}_tiny")

    if args.list_profiles:
        print("Tracking3D recommendation profiles")
        profiles = list_profiles()
        for p in profiles:
            print(f"- {p.key}: {p.title} | {p.summary}")

    if args.recommend is not None:
        try:
            recs = recommend_arches(
                str(args.recommend),
                variant=str(args.variant),
                top_k=int(args.top_k),
            )
        except ValueError as exc:
            print(str(exc))
            print("\nTip: run `python scripts/tracking3d_zoo.py --list-profiles`")
            return 2

        print("Tracking3D recommendations")
        print(f"- profile={str(args.recommend).strip().lower()}")
        print(f"- variant={str(args.variant).strip().lower()}")
        print(f"- top_k={int(args.top_k)}")
        if args.run_smoke_cmds:
            print(f"- rank_by={str(args.rank_by).strip().lower()}")
        print("")

        quiet_run = bool(args.summary_only and args.run_smoke_cmds)
        planned: list[tuple[int, object, str, list[str]]] = []
        for idx, r in enumerate(recs, start=1):
            y = "unknown" if r.year is None else str(r.year)
            if not quiet_run:
                print(f"{idx:02d}. {r.arch_id} | group={r.group} | year={y} | score={r.score:.3f}")
                print(f"    reason: {r.reason}")
            run_name = f"reco_{idx:02d}_{r.family}"
            smoke_cmd = [
                sys.executable,
                "scripts/tracking3d_zoo.py",
                "--smoke",
                str(r.arch_id),
                "--batch-size",
                str(int(args.batch_size)),
                "--seq-len",
                str(int(args.seq_len)),
                "--num-points",
                str(int(args.num_points)),
                "--in-channels",
                str(int(args.in_channels)),
                "--num-classes",
                str(int(args.num_classes)),
                "--width-mult",
                str(float(args.width_mult)),
                "--dropout",
                str(float(args.dropout)),
            ]
            planned.append((idx, r, run_name, smoke_cmd))

        if args.emit_smoke_cmds:
            print("")
            print("Smoke commands")
            for _, _, _, smoke_cmd in planned:
                cmd = "python " + " ".join(smoke_cmd[1:])
                print(f"- {cmd}")

        if args.run_smoke_cmds:
            if not quiet_run:
                print("")
                print("Running smoke commands")
            run_results: list[dict[str, object]] = []
            run_logs: dict[str, dict[str, str]] = {}
            for idx, r, run_name, smoke_cmd in planned:
                t0 = time.perf_counter()
                proc = subprocess.run(
                    smoke_cmd,
                    cwd=str(repo_root),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - t0
                returncode = int(proc.returncode)
                stdout_text = proc.stdout or ""
                stderr_text = proc.stderr or ""
                ok = bool(returncode == 0)
                run_logs[run_name] = {"stdout": stdout_text, "stderr": stderr_text}

                result = {
                    "idx": idx,
                    "arch_id": str(r.arch_id),
                    "family": str(r.family),
                    "group": str(r.group),
                    "year": int(r.year) if r.year is not None else None,
                    "score": float(r.score),
                    "run_name": run_name,
                    "source": "executed",
                    "ok": ok,
                    "returncode": int(returncode),
                    "elapsed_sec": float(elapsed),
                    "stdout_tail": stdout_text.strip().splitlines()[-1]
                    if stdout_text.strip()
                    else "",
                    "stderr_tail": stderr_text.strip().splitlines()[-1]
                    if stderr_text.strip()
                    else "",
                }
                run_results.append(result)

                status = "ok" if ok else f"fail(rc={returncode})"
                if not quiet_run:
                    print(
                        f"- {idx:02d} {r.arch_id} -> {run_name} | {status} | elapsed={elapsed:.2f}s"
                    )
                if not ok and args.fail_fast:
                    if not quiet_run:
                        print("Stopped early due to --fail-fast.")
                    break

            ok_runs = [x for x in run_results if bool(x["ok"])]
            leaderboard_rows: list[dict[str, object]] = []
            print("")
            print("Leaderboard (successful runs)")
            if ok_runs:
                try:
                    ok_runs = _rank_successful_runs(
                        ok_runs, rank_by=str(args.rank_by).strip().lower()
                    )
                except ValueError as exc:
                    print(str(exc))
                    return 2
                for rank, x in enumerate(ok_runs, start=1):
                    print(
                        f"{rank:02d}. {x['arch_id']} | run={x['run_name']} | source={x['source']} | "
                        f"recommend_score={x['score']:.3f} | elapsed={float(x['elapsed_sec']):.2f}s"
                    )
                    row = dict(x)
                    row["rank"] = int(rank)
                    leaderboard_rows.append(row)
            else:
                print("- none")

            if args.save_leaderboard is not None:
                out_path = Path(str(args.save_leaderboard))
                if not out_path.is_absolute():
                    out_path = repo_root / out_path
                try:
                    _save_leaderboard(
                        path=out_path,
                        profile=str(args.recommend).strip().lower(),
                        variant=str(args.variant).strip().lower(),
                        top_k=int(args.top_k),
                        rank_by=str(args.rank_by).strip().lower(),
                        run_results=run_results,
                        leaderboard=leaderboard_rows,
                    )
                except ValueError as exc:
                    print(str(exc))
                    return 2
                print("")
                print(f"Saved leaderboard to {out_path}")

            if args.save_artifacts_dir is not None:
                try:
                    artifacts_path = _resolve_artifacts_dir(
                        repo_root=repo_root,
                        raw=str(args.save_artifacts_dir),
                        profile=str(args.recommend).strip().lower(),
                        variant=str(args.variant).strip().lower(),
                        top_k=int(args.top_k),
                    )
                except ValueError as exc:
                    print(str(exc))
                    return 2
                _save_artifacts(
                    artifacts_dir=artifacts_path,
                    profile=str(args.recommend).strip().lower(),
                    variant=str(args.variant).strip().lower(),
                    top_k=int(args.top_k),
                    rank_by=str(args.rank_by).strip().lower(),
                    planned=planned,
                    run_results=run_results,
                    leaderboard=leaderboard_rows,
                    run_logs=run_logs,
                )
                print("")
                print(f"Saved artifacts to {artifacts_path}")

    if args.smoke is not None:
        import torch

        arch_id = str(args.smoke).strip()
        if ":" not in arch_id:
            arch_id = f"pctrk3d:{arch_id}"
        x = torch.randn(
            int(args.batch_size), int(args.seq_len), int(args.num_points), int(args.in_channels)
        )
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_classes=int(args.num_classes),
            seq_len=int(args.seq_len),
            width_mult=float(args.width_mult),
            dropout=float(args.dropout),
        )
        out = model.track(x)
        print("")
        print(f"smoke: {arch_id}")
        print(f"- output={_summarize(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
