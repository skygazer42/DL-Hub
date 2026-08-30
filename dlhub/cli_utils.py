"""Small presentation helpers shared by repository command-line tools."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import operator
from pathlib import Path


_EXTERNAL_ZOO_PREFIXES = {"timm", "tv", "tvdet", "tvflow", "tvq", "tvseg", "tvvideo"}
_VARIANT_SUFFIXES = ("_tiny", "_small", "_base")


def summarize_output(value: object) -> str:
    """Describe nested model outputs without printing full tensor contents."""

    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        torch = None

    return _summarize_output(value, torch_module=torch, active=set())


def _summarize_output(value: object, *, torch_module: object, active: set[int]) -> str:
    if torch_module is not None and isinstance(value, torch_module.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, dict):
        keys = ", ".join(sorted(map(str, value.keys())))
        return f"dict(keys=[{keys}])"

    if isinstance(value, list | tuple):
        identity = id(value)
        if identity in active:
            return f"<cycle:{type(value).__name__}>"
        active.add(identity)
        try:
            head = ", ".join(
                _summarize_output(item, torch_module=torch_module, active=active)
                for item in value[:2]
            )
        finally:
            active.remove(identity)
        tail = "" if len(value) <= 2 else f", ... (+{len(value) - 2})"
        return f"{type(value).__name__}([{head}{tail}])"

    try:
        logits = getattr(value, "logits")
    except AttributeError:
        return type(value).__name__

    identity = id(value)
    if identity in active:
        return f"<cycle:{type(value).__name__}>"
    active.add(identity)
    try:
        summary = _summarize_output(logits, torch_module=torch_module, active=active)
    finally:
        active.remove(identity)
    return f"{type(value).__name__}(logits={summary})"


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@lru_cache(maxsize=1)
def _zoo_source_index() -> dict[str, tuple[tuple[str, str], ...]]:
    from dlhub.zoo_fidelity import (
        FidelityLevel,
        discover_baseline_wrappers,
        fidelity_for_artifact,
    )

    root = Path(__file__).resolve().parents[1]
    inferred_aliases = {
        wrapper.artifact for wrapper in discover_baseline_wrappers(root)
    }
    rows: dict[str, list[tuple[str, str]]] = {}
    for source_path in sorted((root / "dlhub").rglob("*.py")):
        if source_path.name in {"__init__.py", "_common.py"} or source_path.name.endswith("_zoo.py"):
            continue
        artifact = source_path.relative_to(root).as_posix()
        level = fidelity_for_artifact(artifact)
        if level is FidelityLevel.UNREVIEWED and artifact in inferred_aliases:
            level = FidelityLevel.BASELINE_ALIAS
        rows.setdefault(source_path.stem, []).append((artifact, level.value))
    return {stem: tuple(candidates) for stem, candidates in rows.items()}


def _arch_family(arch_id: str) -> tuple[str, str]:
    prefix, separator, name = str(arch_id).strip().partition(":")
    if not separator:
        return "", prefix
    family = name
    for suffix in _VARIANT_SUFFIXES:
        if family.endswith(suffix):
            family = family[: -len(suffix)]
            break
    return prefix.lower(), family


def format_arch_fidelity(arch_id: str) -> str:
    """Annotate one registered ID without upgrading unresolved source claims."""

    arch = str(arch_id).strip()
    prefix, family = _arch_family(arch)
    if prefix in _EXTERNAL_ZOO_PREFIXES:
        return f"{arch}\tfidelity=external\tsource=external-package"

    source_index = _zoo_source_index()
    candidate_stems = sorted(
        (
            stem
            for stem in source_index
            if family == stem or family.startswith(f"{stem}_")
        ),
        key=len,
        reverse=True,
    )
    if not candidate_stems:
        return f"{arch}\tfidelity=unreviewed\tsource=unresolved"

    longest = len(candidate_stems[0])
    candidates = {
        candidate
        for stem in candidate_stems
        if len(stem) == longest
        for candidate in source_index[stem]
    }
    if len(candidates) != 1:
        levels = {level for _, level in candidates}
        level = levels.pop() if len(levels) == 1 else "unreviewed"
        return f"{arch}\tfidelity={level}\tsource=ambiguous"

    artifact, level = candidates.pop()
    return f"{arch}\tfidelity={level}\tsource={artifact}"


def print_limited(
    lines: Iterable[str],
    *,
    limit: int = 80,
    tail: int = 10,
    annotate_fidelity: bool = False,
) -> None:
    """Print a bounded list while retaining useful entries from both ends."""

    limit = _integer("limit", limit)
    tail = _integer("tail", tail)
    if tail < 0:
        raise ValueError("tail must be >= 0")
    if limit <= 0:
        return
    rows = list(lines)
    if annotate_fidelity:
        rows = [format_arch_fidelity(row) for row in rows]
    if len(rows) <= limit:
        for row in rows:
            print(row)
        return

    tail = min(tail, limit)
    head = limit - tail
    for row in rows[:head]:
        print(row)
    print(f"... ({len(rows) - limit} more) ...")
    if tail:
        for row in rows[-tail:]:
            print(row)


__all__ = ["format_arch_fidelity", "print_limited", "summarize_output"]
