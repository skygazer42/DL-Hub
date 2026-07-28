from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_DIRS = ("dlhub", "tracks", "docs", "scripts", "tests", ".github")
ROOT_FILES = ("LICENSE", "README.md", "Makefile", "mkdocs.yml", "pyproject.toml")
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
RETIRED_CASUAL_LABEL = "to" + "y"
LESSON_REFERENCE = re.compile(
    r"tracks(?P<separator>[/.])(?P<track>[a-z][a-z0-9_]*)"
    r"(?P=separator)(?P<lesson>lesson_[a-z0-9_]+)",
    re.IGNORECASE,
)


def iter_maintained_files() -> Iterable[Path]:
    for relative_dir in MAINTAINED_DIRS:
        base = ROOT / relative_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path

    for relative_file in ROOT_FILES:
        path = ROOT / relative_file
        if path.is_file():
            yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_retired_label(paths: Iterable[Path]) -> list[str]:
    failures: list[str] = []
    needle = RETIRED_CASUAL_LABEL.casefold()

    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if needle in relative.casefold():
            failures.append(f"retired label in path: {relative}")
        if needle in read_text(path).casefold():
            failures.append(f"retired label in content: {relative}")

    return failures


def check_lesson_references(paths: Iterable[Path]) -> list[str]:
    failures: list[str] = []

    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        for match in LESSON_REFERENCE.finditer(read_text(path)):
            track = match.group("track")
            lesson = match.group("lesson")
            if lesson.casefold().startswith("lesson_xx"):
                continue
            lesson_dir = ROOT / "tracks" / track / lesson
            if not lesson_dir.is_dir():
                failures.append(
                    f"missing lesson target in {relative}: tracks/{track}/{lesson}"
                )

    return failures


def main() -> int:
    paths = sorted(set(iter_maintained_files()))
    failures = check_retired_label(paths)
    failures.extend(check_lesson_references(paths))

    if failures:
        print("Narrative contract failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Narrative contract passed ({len(paths)} maintained text files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
