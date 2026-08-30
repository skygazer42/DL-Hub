from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from collections.abc import Iterable
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


def _read_last_jsonl_record(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if ln:
                last = ln
    if not last:
        return None
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


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
            "eval_mean_iou",
            "eval_presence_acc",
            "eval_loss",
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

    if mode == "iou":
        rows.sort(
            key=lambda x: (
                -_num(x.get("eval_mean_iou"), default=-1.0),
                _num(x.get("eval_loss"), default=1e9),
                -_num(x.get("eval_presence_acc"), default=-1.0),
            )
        )
        return rows

    if mode == "acc":
        rows.sort(
            key=lambda x: (
                -_num(x.get("eval_presence_acc"), default=-1.0),
                -_num(x.get("eval_mean_iou"), default=-1.0),
                _num(x.get("eval_loss"), default=1e9),
            )
        )
        return rows

    if mode == "loss":
        rows.sort(
            key=lambda x: (
                _num(x.get("eval_loss"), default=1e9),
                -_num(x.get("eval_mean_iou"), default=-1.0),
                -_num(x.get("eval_presence_acc"), default=-1.0),
            )
        )
        return rows

    raise ValueError(f"Unsupported rank mode: {rank_by!r}. Use one of: iou, acc, loss")


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
    return repo_root / "outputs" / "vision" / "mot_artifacts" / name


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
    lines.append("# MOT Batch Report")
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
    lines.append(
        "| idx | arch_id | run_name | source | ok | eval_iou | eval_acc | eval_loss | elapsed_sec |"
    )
    lines.append("|---:|---|---|---|:---:|---:|---:|---:|---:|")
    for row in run_results:
        lines.append(
            "| {idx} | {arch} | {run} | {source} | {ok} | {iou} | {acc} | {loss} | {elapsed} |".format(
                idx=row.get("idx", ""),
                arch=row.get("arch_id", ""),
                run=row.get("run_name", ""),
                source=row.get("source", ""),
                ok="Y" if bool(row.get("ok")) else "N",
                iou=row.get("eval_mean_iou", ""),
                acc=row.get("eval_presence_acc", ""),
                loss=row.get("eval_loss", ""),
                elapsed=row.get("elapsed_sec", ""),
            )
        )
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    if leaderboard:
        lines.append(
            "| rank | arch_id | run_name | source | eval_iou | eval_acc | eval_loss | elapsed_sec |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")
        for row in leaderboard:
            lines.append(
                "| {rank} | {arch} | {run} | {source} | {iou} | {acc} | {loss} | {elapsed} |".format(
                    rank=row.get("rank", ""),
                    arch=row.get("arch_id", ""),
                    run=row.get("run_name", ""),
                    source=row.get("source", ""),
                    iou=row.get("eval_mean_iou", ""),
                    acc=row.get("eval_presence_acc", ""),
                    loss=row.get("eval_loss", ""),
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

    commands = ["python " + " ".join(train_cmd[1:]) for _, _, _, train_cmd in planned]
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
        description="Vision MOT local model zoo utilities (no downloads)."
    )
    parser.add_argument("--list", action="store_true", help="List available architecture ids.")
    parser.add_argument("--search", type=str, default=None, help="Filter list by substring.")
    parser.add_argument("--limit", type=int, default=80, help="Max lines to print when listing.")
    parser.add_argument("--timeline", action="store_true", help="Print a best-effort MOT timeline.")
    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        metavar="PROFILE",
        help="Recommend architectures for a scenario profile (e.g. realtime, occlusion, long_horizon).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available recommendation profiles and exit.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K recommendations to print with --recommend.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="iou",
        choices=["iou", "acc", "loss"],
        help="Ranking metric for successful runs with --run-train-cmds.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        help="Variant used with --recommend: tiny | small | base.",
    )
    parser.add_argument(
        "--emit-train-cmds",
        action="store_true",
        help="Print runnable lesson_14 training commands for each recommendation.",
    )
    parser.add_argument(
        "--train-device",
        type=str,
        default="cpu",
        help="Device used in emitted training commands (default: cpu).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=2,
        help="Epochs used in emitted training commands.",
    )
    parser.add_argument(
        "--train-max-train-batches",
        type=int,
        default=5,
        help="max-train-batches used in emitted training commands.",
    )
    parser.add_argument(
        "--train-max-eval-batches",
        type=int,
        default=3,
        help="max-eval-batches used in emitted training commands.",
    )
    parser.add_argument(
        "--train-run-prefix",
        type=str,
        default="auto",
        help="Run-name prefix in emitted commands. 'auto' maps to recommendation profile.",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=128,
        help="num-samples used in emitted/runnable training commands.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="batch-size used in emitted/runnable training commands.",
    )
    parser.add_argument(
        "--run-train-cmds",
        action="store_true",
        help="Run emitted lesson_14 commands and print a mini leaderboard.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Used with --run-train-cmds: if run output already exists "
            "(metrics.jsonl found), reuse it instead of retraining."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Used with --run-train-cmds to reduce console output. "
            "Print only final leaderboard and saved paths."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one training command fails (used with --run-train-cmds).",
    )
    parser.add_argument(
        "--save-leaderboard",
        type=str,
        default=None,
        help="Save run results/leaderboard to a file (.json or .csv). Requires --run-train-cmds.",
    )
    parser.add_argument(
        "--save-artifacts-dir",
        type=str,
        default=None,
        help=(
            "Save batch-run artifacts into a directory "
            "(commands, logs, run_results, leaderboard). Use 'auto' for timestamped path. "
            "Requires --run-train-cmds."
        ),
    )
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        metavar="ARCH_ID",
        help="Run a short video smoke on an arch id.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=4, help="Video sequence length.")
    parser.add_argument("--image-size", type=int, default=64, help="Frame spatial size.")
    parser.add_argument("--in-channels", type=int, default=3, help="Frame channels.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes.")
    parser.add_argument("--width-mult", type=float, default=1.0, help="Width multiplier.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout.")
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root_on_path()
    repo_root = Path(__file__).resolve().parents[1]
    from dlhub.vision.mot._timeline import entries
    from dlhub.vision.mot._recommend import list_profiles, recommend_arches
    from dlhub.vision.mot_zoo import build_local_model, list_local_arches

    args = parse_args()
    if args.emit_train_cmds and args.recommend is None:
        print("--emit-train-cmds is only valid with --recommend.")
        print("Tip: python scripts/mot_zoo.py --recommend realtime --emit-train-cmds")
        return 2
    if args.run_train_cmds and args.recommend is None:
        print("--run-train-cmds is only valid with --recommend.")
        print("Tip: python scripts/mot_zoo.py --recommend realtime --run-train-cmds")
        return 2
    if args.skip_existing and not args.run_train_cmds:
        print("--skip-existing is only valid with --run-train-cmds.")
        print(
            "Tip: python scripts/mot_zoo.py --recommend realtime --run-train-cmds --skip-existing"
        )
        return 2
    if args.summary_only and not args.run_train_cmds:
        print("--summary-only is only valid with --run-train-cmds.")
        print("Tip: python scripts/mot_zoo.py --recommend realtime --run-train-cmds --summary-only")
        return 2
    if args.save_leaderboard is not None and not args.run_train_cmds:
        print("--save-leaderboard is only valid with --run-train-cmds.")
        print(
            "Tip: python scripts/mot_zoo.py --recommend realtime --run-train-cmds --save-leaderboard outputs/vision/mot_leaderboard.json"
        )
        return 2
    if args.save_artifacts_dir is not None and not args.run_train_cmds:
        print("--save-artifacts-dir is only valid with --run-train-cmds.")
        print(
            "Tip: python scripts/mot_zoo.py --recommend realtime --run-train-cmds --save-artifacts-dir outputs/vision/mot_artifacts"
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
        print("- python scripts/mot_zoo.py --list")
        print("- python scripts/mot_zoo.py --timeline")
        print("- python scripts/mot_zoo.py --list-profiles")
        print("- python scripts/mot_zoo.py --recommend realtime --top-k 8")
        print("- python scripts/mot_zoo.py --recommend realtime --top-k 8 --emit-train-cmds")
        print("- python scripts/mot_zoo.py --recommend realtime --top-k 3 --run-train-cmds")
        print(
            "- python scripts/mot_zoo.py --recommend realtime --top-k 3 --run-train-cmds --save-artifacts-dir outputs/vision/mot_artifacts"
        )
        print("- python scripts/mot_zoo.py --smoke mot2d:sort_tiny")
        return 2

    arches = list_local_arches()
    needle = str(args.search).lower() if args.search else None
    if needle:
        arches = [a for a in arches if needle in a.lower()]

    if args.list:
        print("Vision MOT local zoo")
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
        print("Vision MOT timeline")
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
            print(f"- {e.family} [{e.group}]: {e.method} -> mot2d:{e.family}_tiny")

    if args.list_profiles:
        print("Vision MOT recommendation profiles")
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
            print("\nTip: run `python scripts/mot_zoo.py --list-profiles`")
            return 2

        print("Vision MOT recommendations")
        print(f"- profile={str(args.recommend).strip().lower()}")
        print(f"- variant={str(args.variant).strip().lower()}")
        print(f"- top_k={int(args.top_k)}")
        if args.run_train_cmds:
            print(f"- rank_by={str(args.rank_by).strip().lower()}")
        print("")
        planned: list[tuple[int, object, str, list[str]]] = []
        profile = str(args.recommend).strip().lower()
        run_prefix = (
            profile
            if str(args.train_run_prefix).strip().lower() == "auto"
            else str(args.train_run_prefix).strip()
        )
        quiet_run = bool(args.summary_only and args.run_train_cmds)
        for idx, r in enumerate(recs, start=1):
            y = "unknown" if r.year is None else str(r.year)
            if not quiet_run:
                print(f"{idx:02d}. {r.arch_id} | group={r.group} | year={y} | score={r.score:.3f}")
                print(f"    reason: {r.reason}")
            run_name = f"{run_prefix}_{idx:02d}_{r.family}"
            train_cmd = [
                sys.executable,
                "-m",
                "tracks.vision.lesson_14_video_mot_basics.train",
                "--arch",
                str(r.arch_id),
                "--device",
                str(args.train_device).strip(),
                "--epochs",
                str(int(args.train_epochs)),
                "--max-train-batches",
                str(int(args.train_max_train_batches)),
                "--max-eval-batches",
                str(int(args.train_max_eval_batches)),
                "--num-samples",
                str(int(args.train_num_samples)),
                "--batch-size",
                str(int(args.train_batch_size)),
                "--run-name",
                run_name,
            ]
            planned.append((idx, r, run_name, train_cmd))

        if args.emit_train_cmds:
            print("")
            print("Training commands (lesson_14)")
            for _, _, _, train_cmd in planned:
                cmd = "python " + " ".join(train_cmd[1:])
                print(f"- {cmd}")

        if args.run_train_cmds:
            if not quiet_run:
                print("")
                print("Running training commands (lesson_14)")
            run_results: list[dict[str, object]] = []
            run_logs: dict[str, dict[str, str]] = {}
            for idx, r, run_name, train_cmd in planned:
                run_dir = repo_root / "outputs" / "vision" / "lesson_14_video_mot_basics" / run_name
                used_existing = False
                metrics = {}
                stdout_text = ""
                stderr_text = ""
                elapsed = 0.0
                returncode = 0

                existing = _read_last_jsonl_record(run_dir / "metrics.jsonl")
                if args.skip_existing and existing is not None:
                    used_existing = True
                    metrics = existing
                else:
                    t0 = time.perf_counter()
                    proc = subprocess.run(
                        train_cmd,
                        cwd=str(repo_root),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    elapsed = time.perf_counter() - t0
                    returncode = int(proc.returncode)
                    stdout_text = proc.stdout or ""
                    stderr_text = proc.stderr or ""
                    metrics = _read_last_jsonl_record(run_dir / "metrics.jsonl") or {}

                eval_iou = metrics.get("eval_mean_iou")
                eval_acc = metrics.get("eval_presence_acc")
                eval_loss = metrics.get("eval_loss")
                ok = bool(used_existing or returncode == 0)
                run_logs[run_name] = {"stdout": stdout_text, "stderr": stderr_text}
                result = {
                    "idx": idx,
                    "arch_id": str(r.arch_id),
                    "family": str(r.family),
                    "group": str(r.group),
                    "year": int(r.year) if r.year is not None else None,
                    "score": float(r.score),
                    "run_name": run_name,
                    "source": "existing" if used_existing else "executed",
                    "ok": ok,
                    "returncode": int(returncode),
                    "elapsed_sec": float(elapsed),
                    "eval_loss": eval_loss,
                    "eval_presence_acc": eval_acc,
                    "eval_mean_iou": eval_iou,
                    "stdout_tail": stdout_text.strip().splitlines()[-1]
                    if stdout_text.strip()
                    else "",
                    "stderr_tail": stderr_text.strip().splitlines()[-1]
                    if stderr_text.strip()
                    else "",
                }
                run_results.append(result)
                status = "existing" if used_existing else ("ok" if ok else f"fail(rc={returncode})")
                if not quiet_run:
                    print(
                        f"- {idx:02d} {r.arch_id} -> {run_name} | {status} | "
                        f"elapsed={elapsed:.2f}s | eval_iou={eval_iou} | eval_acc={eval_acc}"
                    )
                if not ok and args.fail_fast:
                    if not quiet_run:
                        print("Stopped early due to --fail-fast.")
                    break

            ok_runs = [x for x in run_results if bool(x["ok"])]
            leaderboard_rows: list[dict[str, object]] = []
            print("")
            if ok_runs:
                try:
                    ok_runs = _rank_successful_runs(
                        ok_runs, rank_by=str(args.rank_by).strip().lower()
                    )
                except ValueError as exc:
                    print(str(exc))
                    return 2
                print("Leaderboard (successful runs)")
                for rank, x in enumerate(ok_runs, start=1):
                    print(
                        f"{rank:02d}. {x['arch_id']} | run={x['run_name']} | "
                        f"source={x['source']} | "
                        f"eval_iou={x['eval_mean_iou']} | eval_acc={x['eval_presence_acc']} | "
                        f"eval_loss={x['eval_loss']} | elapsed={float(x['elapsed_sec']):.2f}s"
                    )
                    row = dict(x)
                    row["rank"] = int(rank)
                    leaderboard_rows.append(row)
            else:
                print("Leaderboard (successful runs)")
                print("- none")

            if args.save_leaderboard is not None:
                out_path = Path(str(args.save_leaderboard))
                if not out_path.is_absolute():
                    out_path = repo_root / out_path
                try:
                    _save_leaderboard(
                        path=out_path,
                        profile=profile,
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
                        profile=profile,
                        variant=str(args.variant).strip().lower(),
                        top_k=int(args.top_k),
                    )
                except ValueError as exc:
                    print(str(exc))
                    return 2
                _save_artifacts(
                    artifacts_dir=artifacts_path,
                    profile=profile,
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
            arch_id = f"mot2d:{arch_id}"

        x = torch.randn(
            int(args.batch_size),
            int(args.seq_len),
            int(args.in_channels),
            int(args.image_size),
            int(args.image_size),
        )
        model = build_local_model(
            arch_id,
            in_channels=int(args.in_channels),
            num_classes=int(args.num_classes),
            seq_len=int(args.seq_len),
            image_size=int(args.image_size),
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
