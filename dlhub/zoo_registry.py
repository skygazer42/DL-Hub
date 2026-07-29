"""Shared primitives for local Model Zoo registries.

Task-specific Zoo modules should describe families and build arguments.  This
module owns architecture-id parsing and the repeated lazy-import closure so the
same edge cases are not reimplemented across every task.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from typing import TypeVar

ConfigT = TypeVar("ConfigT")
Builder = Callable[[ConfigT], object]
KwargsFactory = Callable[[ConfigT, str], Mapping[str, object]]


def split_arch_id(
    arch_id: str,
    *,
    default_prefix: str | None = None,
    example: str | None = None,
) -> tuple[str, str]:
    """Normalize ``prefix:name`` ids and optionally accept a bare name."""

    normalized = str(arch_id).strip()
    if ":" not in normalized:
        if default_prefix is None:
            example_text = f" like {example!r}" if example else ""
            raise ValueError(
                f"Expected a namespaced arch id{example_text}, got: {normalized!r}"
            )
        prefix = str(default_prefix).strip().lower()
        name = normalized
    else:
        prefix, name = normalized.split(":", 1)
        prefix = prefix.strip().lower()
        name = name.strip()

    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {normalized!r}")
    return prefix, name


def list_arch_ids(registry: Mapping[str, object], *, prefix: str) -> list[str]:
    """Return deterministic namespaced ids for a registry mapping."""

    namespace = str(prefix).strip().lower().rstrip(":")
    if not namespace:
        raise ValueError("prefix must not be empty")
    return [f"{namespace}:{name}" for name in sorted(registry)]


def make_lazy_family_registry(
    families: Iterable[str],
    sizes: Iterable[str],
    *,
    module_template: str,
    builder_template: str,
    kwargs_factory: KwargsFactory[ConfigT],
) -> dict[str, Builder[ConfigT]]:
    """Build a family/size registry whose implementation modules load on demand."""

    registry: dict[str, Builder[ConfigT]] = {}
    for family_value in families:
        family = str(family_value).strip()
        if not family:
            raise ValueError("family names must not be empty")
        for size_value in sizes:
            size = str(size_value).strip()
            if not size:
                raise ValueError("size names must not be empty")
            variant = f"{family}_{size}"
            if variant in registry:
                raise ValueError(f"duplicate Zoo variant: {variant}")

            def _builder(
                cfg: ConfigT,
                family: str = family,
                variant: str = variant,
            ) -> object:
                module_name = module_template.format(family=family)
                builder_name = builder_template.format(family=family)
                module = importlib.import_module(module_name)
                fn = getattr(module, builder_name, None)
                if not callable(fn):
                    raise RuntimeError(f"Zoo module {module_name!r} missing {builder_name}()")
                return fn(**dict(kwargs_factory(cfg, variant)))

            registry[variant] = _builder
    return registry


__all__ = ["list_arch_ids", "make_lazy_family_registry", "split_arch_id"]
