from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_TRAIN_FLAGS = frozenset({"--device", "--run-name", "--seed"})
OFFLINE_BUILT_IN = "built-in"
OFFLINE_EXPLICIT_FAKE = "explicit-fake"
OFFLINE_EXTERNAL_ONLY = "external-only"


@dataclass(frozen=True)
class LessonContract:
    track: str
    lesson: str
    entrypoint: str | None
    entrypoint_module: str | None
    entrypoint_kind: str | None
    cli_flags: tuple[str, ...]
    offline_mode: str
    uses_output_layout: bool
    has_init: bool
    has_model: bool
    has_data: bool
    has_readme: bool
    parse_error: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.track, self.lesson


@dataclass(frozen=True)
class DocumentedLessonCommand:
    path: str
    line: int
    module: str
    flags: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _argument_flags(node: ast.Call) -> set[str]:
    if _call_name(node) != "add_argument":
        return set()
    return {
        arg.value
        for arg in node.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }


def _literal_strings(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _inspect_lesson(lesson_dir: Path, root: Path) -> LessonContract:
    candidates = [
        path for path in (lesson_dir / "train.py", lesson_dir / "run.py") if path.is_file()
    ]
    entrypoint_path = candidates[0] if candidates else None
    entrypoint = str(entrypoint_path.relative_to(root)) if entrypoint_path is not None else None
    entrypoint_kind = entrypoint_path.stem if entrypoint_path is not None else None
    entrypoint_module = (
        ".".join(entrypoint_path.relative_to(root).with_suffix("").parts)
        if entrypoint_path is not None
        else None
    )

    flags: set[str] = set()
    dataset_literals: set[str] = set()
    uses_output_layout = False
    parse_error = None
    if entrypoint_path is not None:
        try:
            tree = ast.parse(
                entrypoint_path.read_text(encoding="utf-8"), filename=str(entrypoint_path)
            )
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_error = str(exc)
        else:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                uses_output_layout |= _call_name(node) == "build_run_paths"
                call_flags = _argument_flags(node)
                flags.update(call_flags)
                if "--dataset" in call_flags:
                    dataset_literals.update(_literal_strings(node))

    if "--dataset" not in flags:
        offline_mode = OFFLINE_BUILT_IN
    elif any(re.search(r"\bfake\b", literal) for literal in dataset_literals):
        offline_mode = OFFLINE_EXPLICIT_FAKE
    else:
        offline_mode = OFFLINE_EXTERNAL_ONLY

    return LessonContract(
        track=lesson_dir.parent.name,
        lesson=lesson_dir.name,
        entrypoint=entrypoint,
        entrypoint_module=entrypoint_module,
        entrypoint_kind=entrypoint_kind,
        cli_flags=tuple(sorted(flags)),
        offline_mode=offline_mode,
        uses_output_layout=uses_output_layout,
        has_init=(lesson_dir / "__init__.py").is_file(),
        has_model=(lesson_dir / "model.py").is_file(),
        has_data=(lesson_dir / "data.py").is_file(),
        has_readme=(lesson_dir / "README.md").is_file(),
        parse_error=parse_error,
    )


def discover_lesson_contracts(root: Path | None = None) -> list[LessonContract]:
    root = root or repo_root()
    tracks_dir = root / "tracks"
    contracts: list[LessonContract] = []
    for track_dir in sorted(tracks_dir.iterdir()):
        if not track_dir.is_dir() or track_dir.name.startswith("__"):
            continue
        for lesson_dir in sorted(track_dir.glob("lesson_*")):
            if lesson_dir.is_dir():
                contracts.append(_inspect_lesson(lesson_dir, root))
    return contracts


def get_lesson_contract(track: str, lesson: str, root: Path | None = None) -> LessonContract | None:
    root = root or repo_root()
    lesson_dir = root / "tracks" / track / lesson
    if not lesson_dir.is_dir():
        return None
    return _inspect_lesson(lesson_dir, root)


def discover_curated_smoke_lessons(root: Path | None = None) -> set[tuple[str, str]]:
    root = root or repo_root()
    lessons: set[tuple[str, str]] = set()
    for path in sorted((root / "scripts" / "smoke_checks").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            parts = module.split(".")
            if len(parts) >= 3 and parts[0] == "tracks" and parts[2].startswith("lesson_"):
                lessons.add((parts[1], parts[2]))
    return lessons


def _active_markdown_paths(root: Path) -> list[Path]:
    paths = [root / "README.md"]
    for path in (root / "docs").rglob("*.md"):
        relative = path.relative_to(root / "docs")
        if relative.parts[0] in {"plans", "superpowers"}:
            continue
        paths.append(path)
    paths.extend((root / "tracks").glob("**/*.md"))
    return sorted(set(paths))


def discover_documented_lesson_commands(
    root: Path | None = None,
) -> list[DocumentedLessonCommand]:
    root = root or repo_root()
    commands: list[DocumentedLessonCommand] = []
    command_start = re.compile(r"^python(?:3)?\s+-m\s+tracks\.")
    for path in _active_markdown_paths(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        line_index = 0
        while line_index < len(lines):
            stripped = lines[line_index].strip()
            if not command_start.match(stripped):
                line_index += 1
                continue
            start_line = line_index + 1
            parts: list[str] = []
            while True:
                current = lines[line_index].strip()
                continued = current.endswith("\\")
                parts.append(current[:-1].strip() if continued else current)
                line_index += 1
                if not continued or line_index >= len(lines):
                    break
            try:
                tokens = shlex.split(" ".join(parts))
            except ValueError:
                continue
            if len(tokens) < 3 or tokens[1] != "-m":
                continue
            module = tokens[2]
            if "lesson_XX" in module:
                continue
            flags = tuple(
                sorted(
                    {
                        token.split("=", 1)[0]
                        for token in tokens[3:]
                        if token.startswith("--") and token != "--"
                    }
                )
            )
            commands.append(
                DocumentedLessonCommand(
                    path=str(path.relative_to(root)),
                    line=start_line,
                    module=module,
                    flags=flags,
                )
            )
    return commands


def validate_lesson_contracts(root: Path | None = None) -> list[str]:
    root = root or repo_root()
    contracts = discover_lesson_contracts(root)
    by_key = {contract.key: contract for contract in contracts}
    errors: list[str] = []

    for contract in contracts:
        label = f"{contract.track}/{contract.lesson}"
        lesson_dir = root / "tracks" / contract.track / contract.lesson
        entrypoints = [
            path.name for path in (lesson_dir / "train.py", lesson_dir / "run.py") if path.is_file()
        ]
        if len(entrypoints) != 1:
            errors.append(
                f"{label}: expected exactly one train.py/run.py entrypoint, found {entrypoints}"
            )
        if not contract.has_init:
            errors.append(f"{label}: missing __init__.py")
        if contract.parse_error:
            errors.append(f"{label}: entrypoint cannot be parsed: {contract.parse_error}")
            continue
        if contract.entrypoint_kind == "train":
            missing = sorted(REQUIRED_TRAIN_FLAGS.difference(contract.cli_flags))
            if missing:
                errors.append(
                    f"{label}: training entrypoint is missing core flags: {', '.join(missing)}"
                )
            if not contract.uses_output_layout:
                errors.append(f"{label}: training entrypoint does not use build_run_paths")
        if contract.offline_mode == OFFLINE_EXTERNAL_ONLY:
            errors.append(f"{label}: --dataset is exposed without an explicit fake option")

    smoke_lessons = discover_curated_smoke_lessons(root)
    unknown_smoke_lessons = sorted(smoke_lessons.difference(by_key))
    for track, lesson in unknown_smoke_lessons:
        errors.append(f"smoke suite references an unknown lesson: {track}/{lesson}")
    all_tracks = {contract.track for contract in contracts}
    covered_tracks = {track for track, _ in smoke_lessons}
    for track in sorted(all_tracks.difference(covered_tracks)):
        errors.append(f"curated smoke suite has no lesson from track: {track}")

    for command in discover_documented_lesson_commands(root):
        parts = command.module.split(".")
        if len(parts) < 4 or parts[0] != "tracks":
            continue
        contract = by_key.get((parts[1], parts[2]))
        label = f"{command.path}:{command.line}"
        if contract is None:
            errors.append(f"{label}: documents unknown lesson module {command.module}")
            continue
        if command.module != contract.entrypoint_module:
            errors.append(
                f"{label}: documents {command.module}, expected {contract.entrypoint_module}"
            )
        unsupported = sorted(set(command.flags).difference(contract.cli_flags))
        if unsupported:
            errors.append(f"{label}: {command.module} does not support {', '.join(unsupported)}")

    return errors


def format_summary(contracts: list[LessonContract], root: Path | None = None) -> str:
    root = root or repo_root()
    smoke_lessons = discover_curated_smoke_lessons(root)
    tracks = {contract.track for contract in contracts}
    covered_tracks = {track for track, _ in smoke_lessons}
    explicit_fake = sum(contract.offline_mode == OFFLINE_EXPLICIT_FAKE for contract in contracts)
    built_in = sum(contract.offline_mode == OFFLINE_BUILT_IN for contract in contracts)
    return "\n".join(
        [
            f"lesson contracts: {len(contracts)} lessons across {len(tracks)} tracks",
            f"- entrypoints: {sum(contract.entrypoint_kind == 'train' for contract in contracts)} train, "
            f"{sum(contract.entrypoint_kind == 'run' for contract in contracts)} run",
            f"- offline interfaces: {explicit_fake} explicit --dataset fake, {built_in} built-in data",
            f"- optional structure gaps: {sum(not contract.has_model for contract in contracts)} model.py, "
            f"{sum(not contract.has_data for contract in contracts)} data.py, "
            f"{sum(not contract.has_readme for contract in contracts)} README.md",
            f"- curated smoke: {len(smoke_lessons)}/{len(contracts)} lessons, "
            f"{len(covered_tracks)}/{len(tracks)} tracks",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and validate DL-Hub lesson entrypoints without importing lesson modules."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate lesson and documentation contracts."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the complete contract inventory as JSON."
    )
    args = parser.parse_args(argv)

    root = repo_root()
    contracts = discover_lesson_contracts(root)
    if args.json:
        print(
            json.dumps([asdict(contract) for contract in contracts], ensure_ascii=False, indent=2)
        )
    else:
        print(format_summary(contracts, root))

    if not args.check:
        return 0
    errors = validate_lesson_contracts(root)
    if errors:
        print(f"lesson contracts: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"- {error}")
        return 1
    print("lesson contracts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
