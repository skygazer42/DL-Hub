import argparse
import difflib
import keyword
import shlex
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tracks_dir() -> Path:
    return _repo_root() / "tracks"


def _is_module_component(value: str) -> bool:
    """Return whether *value* can safely be used as one dotted-module component."""

    return value.isidentifier() and not keyword.iskeyword(value)


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
        if _is_module_component(child.name) and (child / "__init__.py").is_file():
            tracks.append(child.name)
    return tracks


def discover_lessons(track: str) -> list[str]:
    if not _is_module_component(track):
        return []
    track_dir = _tracks_dir() / track
    if not track_dir.is_dir():
        return []

    lessons: list[str] = []
    for child in sorted(track_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("lesson_"):
            continue
        if not _is_module_component(child.name):
            continue
        if (child / "train.py").is_file() or (child / "run.py").is_file():
            lessons.append(child.name)
    return lessons


def resolve_entrypoint_module(track: str, lesson: str) -> str:
    if not _is_module_component(track):
        raise FileNotFoundError(f"Invalid track name: {track!r}")
    if not lesson.startswith("lesson_") or not _is_module_component(lesson):
        raise FileNotFoundError(f"Invalid lesson name: {lesson!r}")

    lesson_dir = _tracks_dir() / track / lesson
    if not lesson_dir.is_dir():
        raise FileNotFoundError(f"Lesson not found: tracks/{track}/{lesson}")

    candidates = [name for name in ("train", "run") if (lesson_dir / f"{name}.py").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No entrypoint found (train.py/run.py): tracks/{track}/{lesson}")
    if len(candidates) > 1:
        names = ", ".join(f"{name}.py" for name in candidates)
        raise FileNotFoundError(
            f"Ambiguous entrypoint for tracks/{track}/{lesson}: found {names}; keep exactly one"
        )
    return f"tracks.{track}.{lesson}.{candidates[0]}"


def _print_tracks() -> bool:
    tracks = discover_tracks()
    if not tracks:
        print("No tracks found.")
        return False

    print("Tracks:")
    for t in tracks:
        print(f"- {t}")
    return True


def _print_lessons(track: str) -> bool:
    lessons = discover_lessons(track)
    if not lessons:
        print(f"No lessons found for track: {track}")
        return False

    print(f"Lessons ({track}):")
    for lesson_name in lessons:
        print(f"- {lesson_name}")
    return True


def _split_forwarded_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split launcher arguments from the explicit ``--`` pass-through tail."""

    try:
        separator = argv.index("--")
    except ValueError:
        return argv, []
    return argv[:separator], argv[separator + 1 :]


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _closest(value: str, choices: list[str] | tuple[str, ...]) -> str | None:
    matches = difflib.get_close_matches(value, choices, n=1, cutoff=0.55)
    return matches[0] if matches else None


def _print_resolution_help(track: str, lesson: str, message: str) -> None:
    print(message, file=sys.stderr)
    tracks = discover_tracks()
    if track not in tracks:
        suggestion = _closest(track, tracks)
        if suggestion:
            print(f"Did you mean track {suggestion!r}?", file=sys.stderr)
        print("Run with --list to see available tracks.", file=sys.stderr)
        return

    suggestion = _closest(lesson, discover_lessons(track))
    if suggestion:
        print(f"Did you mean lesson {suggestion!r}?", file=sys.stderr)
    print(f"Run with {track} --list to see available lessons.", file=sys.stderr)


def _get_lesson_contract(track: str, lesson: str):
    # Support both ``python scripts/run_lesson.py`` and
    # ``python -m scripts.run_lesson`` / normal package imports.
    if __package__:
        from .lesson_contracts import get_lesson_contract
    else:
        from lesson_contracts import get_lesson_contract

    return get_lesson_contract(track, lesson)


def _validate_implicit_lesson_args(
    parser: argparse.ArgumentParser,
    forwarded: list[str],
    *,
    supported_flags: tuple[str, ...],
) -> None:
    """Catch option typos while retaining the historical no-separator syntax.

    Explicit arguments after ``--`` are intentionally not inspected.  Before
    the separator, long options must be statically declared by the lesson.
    """

    supported = set(supported_flags)
    launcher_flags = {"--list", "--dry-run", "--describe", "--help"}
    for token in forwarded:
        if not token.startswith("--") or token == "--":
            continue
        option = _option_name(token)
        if option in supported:
            continue
        suggestion = _closest(option, tuple(sorted(supported | launcher_flags)))
        hint = f" Did you mean {suggestion!r}?" if suggestion else ""
        parser.error(
            f"unrecognized option {option!r}.{hint} "
            "Use '--' before lesson arguments to forward them verbatim."
        )


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
        allow_abbrev=False,
    )
    parser.add_argument("track", nargs="?", help="Track name under tracks/ (e.g. vision, nlp, gnn)")
    parser.add_argument(
        "lesson", nargs="?", help="Lesson directory name (e.g. lesson_01_mnist_lenet)"
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list", action="store_true", help="List available tracks or lessons.")
    actions.add_argument(
        "--dry-run", action="store_true", help="Print the resolved module without running it."
    )
    actions.add_argument(
        "--describe",
        action="store_true",
        help="Show the lesson's supported CLI flags and data mode.",
    )

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    launcher_argv, explicit_forwarded = _split_forwarded_args(raw_argv)
    args, implicit_forwarded = parser.parse_known_args(launcher_argv)

    if args.list:
        if args.lesson:
            parser.error("--list accepts at most a track name, not a lesson name")
        if implicit_forwarded or explicit_forwarded:
            parser.error("--list does not accept lesson arguments")
        if args.track and not args.lesson:
            if args.track not in discover_tracks():
                _print_resolution_help(args.track, "", f"Track not found: tracks/{args.track}")
                return 2
            return 0 if _print_lessons(args.track) else 2
        return 0 if _print_tracks() else 2

    if not args.track:
        if implicit_forwarded or explicit_forwarded:
            parser.error("lesson arguments require both a track and lesson name")
        parser.print_help()
        print("\nTip: add --list to discover tracks/lessons.")
        return 2
    if not args.lesson:
        if implicit_forwarded or explicit_forwarded:
            parser.error("lesson arguments require a lesson name")
        _print_lessons(args.track)
        print("\nTip: provide a lesson name to run it.")
        return 2

    try:
        module = resolve_entrypoint_module(args.track, args.lesson)
    except FileNotFoundError as exc:
        _print_resolution_help(args.track, args.lesson, str(exc))
        return 2

    contract = _get_lesson_contract(args.track, args.lesson)
    if contract is None:
        print(f"Lesson contract not found: {args.track}/{args.lesson}", file=sys.stderr)
        return 2

    if args.describe:
        if implicit_forwarded or explicit_forwarded:
            parser.error("--describe does not accept lesson arguments")
        if __package__:
            from .lesson_contracts import OFFLINE_EXPLICIT_FAKE
        else:
            from lesson_contracts import OFFLINE_EXPLICIT_FAKE

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

    _validate_implicit_lesson_args(
        parser,
        implicit_forwarded,
        supported_flags=contract.cli_flags,
    )
    lesson_args = [*implicit_forwarded, *explicit_forwarded]

    cmd = [sys.executable, "-m", module, *lesson_args]
    print("Running:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(_repo_root()), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
