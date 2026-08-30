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
ZOO_OVERCLAIM_PATTERNS = (
    (re.compile(r"\bArchitecture IDs?\b", re.IGNORECASE), "use registration ID"),
    (re.compile(r"算法族"), "use 方法标签 or 注册组"),
    (re.compile(r"架构族"), "use 方法标签 or 注册组"),
    (re.compile(r"完整的生成模型架构库"), "describe the registration timeline"),
    (
        re.compile(r"所有(?:实现|\s*backbone)均为纯 PyTorch", re.IGNORECASE),
        "separate local source format from implementation fidelity",
    ),
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


def _is_core_zoo_page(relative: Path) -> bool:
    if relative.as_posix() in {"README.md", "docs/index.md"}:
        return True
    return (
        len(relative.parts) >= 3
        and relative.parts[0] == "docs"
        and relative.parts[1] in {"tracks", "zoo"}
        and relative.suffix == ".md"
    )


def check_zoo_claims(paths: Iterable[Path], *, root: Path = ROOT) -> list[str]:
    """Keep registration counts distinct from implementation/fidelity claims."""

    failures: list[str] = []
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not _is_core_zoo_page(relative):
            continue
        text = read_text(path)
        for pattern, replacement in ZOO_OVERCLAIM_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"ambiguous Zoo claim in {relative.as_posix()}:{line}: "
                    f"{match.group(0)!r}; {replacement}"
                )
    return failures


def main() -> int:
    paths = sorted(set(iter_maintained_files()))
    failures = check_retired_label(paths)
    failures.extend(check_lesson_references(paths))
    failures.extend(check_zoo_claims(paths))

    if failures:
        print("Narrative contract failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Narrative contract passed ({len(paths)} maintained text files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
