import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tracks_dir() -> Path:
    return _repo_root() / "tracks"


def discover_tracks() -> list[str]:
    tracks_dir = _tracks_dir()
    if not tracks_dir.is_dir():
        return []

    tracks: list[str] = []
    for child in sorted(tracks_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("__"):
            continue
        if (child / "__init__.py").is_file():
            tracks.append(child.name)
    return tracks


def discover_lessons(track: str) -> list[str]:
    track_dir = _tracks_dir() / track
    if not track_dir.is_dir():
        return []

    lessons: list[str] = []
    for child in sorted(track_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("lesson_"):
            continue
        if (child / "train.py").is_file() or (child / "run.py").is_file():
            lessons.append(child.name)
    return lessons


def resolve_entrypoint_module(track: str, lesson: str) -> str:
    lesson_dir = _tracks_dir() / track / lesson
    if not lesson_dir.is_dir():
        raise FileNotFoundError(f"Lesson not found: tracks/{track}/{lesson}")

    if (lesson_dir / "train.py").is_file():
        return f"tracks.{track}.{lesson}.train"
    if (lesson_dir / "run.py").is_file():
        return f"tracks.{track}.{lesson}.run"
    raise FileNotFoundError(f"No entrypoint found (train.py/run.py): tracks/{track}/{lesson}")


def _print_tracks() -> None:
    tracks = discover_tracks()
    if not tracks:
        print("No tracks found.")
        return

    print("Tracks:")
    for t in tracks:
        print(f"- {t}")


def _print_lessons(track: str) -> None:
    lessons = discover_lessons(track)
    if not lessons:
        print(f"No lessons found for track: {track}")
        return

    print(f"Lessons ({track}):")
    for lesson_name in lessons:
        print(f"- {lesson_name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and run lesson entrypoints from one launcher.\n"
            "Arguments after the lesson name are forwarded to that lesson's own CLI.\n\n"
            "Examples:\n"
            "  python scripts/run_lesson.py --list\n"
            "  python scripts/run_lesson.py vision --list\n"
            "  python scripts/run_lesson.py vision lesson_06_swin_compact_classification --describe\n"
            "  python scripts/run_lesson.py vision lesson_06_swin_compact_classification -- --device cpu --epochs 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("track", nargs="?", help="Track name under tracks/ (e.g. vision, nlp, gnn)")
    parser.add_argument(
        "lesson", nargs="?", help="Lesson directory name (e.g. lesson_01_mnist_lenet)"
    )
    parser.add_argument("--list", action="store_true", help="List available tracks or lessons.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved module without running it."
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Show the lesson's supported CLI flags and data mode.",
    )

    args, unknown = parser.parse_known_args(argv)

    if args.list:
        if args.track and not args.lesson:
            _print_lessons(args.track)
            return 0
        _print_tracks()
        return 0

    if not args.track:
        parser.print_help()
        print("\nTip: add --list to discover tracks/lessons.")
        return 2
    if not args.lesson:
        _print_lessons(args.track)
        print("\nTip: provide a lesson name to run it.")
        return 2

    try:
        module = resolve_entrypoint_module(args.track, args.lesson)
    except FileNotFoundError as exc:
        print(str(exc))
        if args.track not in discover_tracks():
            print("\nTip: available tracks:")
            _print_tracks()
        else:
            print("\nTip: available lessons:")
            _print_lessons(args.track)
        return 2

    if args.describe:
        from lesson_contracts import OFFLINE_EXPLICIT_FAKE, get_lesson_contract

        contract = get_lesson_contract(args.track, args.lesson)
        if contract is None:
            print(f"Lesson contract not found: {args.track}/{args.lesson}")
            return 2
        print(f"Lesson: {contract.track}/{contract.lesson}")
        print(f"Entrypoint: {contract.entrypoint_module}")
        if contract.offline_mode == OFFLINE_EXPLICIT_FAKE:
            print("Offline data: pass --dataset fake")
        else:
            print("Offline data: built in (no --dataset argument needed)")
        print("CLI flags:")
        for flag in contract.cli_flags:
            print(f"- {flag}")
        return 0

    if unknown and unknown[0] == "--":
        unknown = unknown[1:]

    cmd = [sys.executable, "-m", module, *unknown]
    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(_repo_root()), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
