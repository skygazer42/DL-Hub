"""Deterministic, offline integrity checks for every local ``*_zoo.py`` registry."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import re
import socket
import sys
import time
import tokenize
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

ARCH_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*:[a-z0-9][a-z0-9_.-]*\Z")
_BOUND_NAME_PATTERN = re.compile(r"[^a-z0-9]")


class UnexpectedNetworkAccess(RuntimeError):
    """Raised when a Zoo import or representative build attempts network I/O."""


@dataclass(frozen=True)
class ModuleAudit:
    module: str
    path: str
    parsed: bool
    imported: bool
    id_count: int
    namespaces: tuple[str, ...]
    registry_count: int
    builder_count: int
    binding_checked: int


@dataclass(frozen=True)
class SmokeAudit:
    domain: str
    module: str
    arch_id: str
    operation: str
    tensor_count: int
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class ZooAudit:
    modules: tuple[ModuleAudit, ...]
    errors: tuple[str, ...]

    @property
    def summary(self) -> dict[str, int | bool]:
        return {
            "modules_discovered": len(self.modules),
            "modules_parsed": sum(module.parsed for module in self.modules),
            "modules_imported": sum(module.imported for module in self.modules),
            "namespaces": len(
                {namespace for module in self.modules for namespace in module.namespaces}
            ),
            "ids": sum(module.id_count for module in self.modules),
            "registries": sum(module.registry_count for module in self.modules),
            "builders": sum(module.builder_count for module in self.modules),
            "bindings_checked": sum(module.binding_checked for module in self.modules),
            "network_blocked": True,
            "ok": not self.errors,
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path(root: Path) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def discover_zoo_paths(root: Path | None = None) -> list[Path]:
    root = root or repo_root()
    return sorted((root / "dlhub").rglob("*_zoo.py"), key=lambda path: path.as_posix())


def _module_name(path: Path, root: Path) -> str:
    return ".".join(path.relative_to(root).with_suffix("").parts)


def _read_python(path: Path) -> str:
    with tokenize.open(path) as source_file:
        return source_file.read()


def _blocked_network(operation: str):
    def blocked(*args: object, **kwargs: object) -> None:
        del kwargs
        target = repr(args[:2]) if args else "unknown target"
        raise UnexpectedNetworkAccess(f"blocked {operation}: {target}")

    return blocked


@contextmanager
def block_network() -> Iterator[None]:
    """Block common socket and model-download paths during Zoo checks."""

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_MODE": "offline",
                },
            )
        )
        stack.enter_context(
            mock.patch.object(
                socket.socket,
                "connect",
                _blocked_network("socket.connect"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                socket,
                "create_connection",
                _blocked_network("socket.create_connection"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                urllib.request,
                "urlopen",
                _blocked_network("urllib.request.urlopen"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                urllib.request,
                "urlretrieve",
                _blocked_network("urllib.request.urlretrieve"),
            )
        )

        import torch
        import torch.utils.model_zoo as torch_model_zoo

        stack.enter_context(
            mock.patch.object(
                torch.hub,
                "download_url_to_file",
                _blocked_network("torch.hub.download_url_to_file"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch.hub,
                "load_state_dict_from_url",
                _blocked_network("torch.hub.load_state_dict_from_url"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                torch_model_zoo,
                "load_url",
                _blocked_network("torch.utils.model_zoo.load_url"),
            )
        )

        try:
            import requests
        except ImportError:
            pass
        else:
            stack.enter_context(
                mock.patch.object(
                    requests.sessions.Session,
                    "request",
                    _blocked_network("requests.Session.request"),
                )
            )
        yield


def _top_level_functions(tree: ast.Module) -> set[str]:
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _normalize_bound_name(value: str) -> str:
    return _BOUND_NAME_PATTERN.sub("", value.lower())


def _bound_variants(builder: object) -> tuple[str, ...]:
    if not callable(builder):
        return ()
    values: list[str] = []
    try:
        closure = inspect.getclosurevars(builder).nonlocals
    except TypeError:
        closure = {}
    for name in ("variant", "v"):
        value = closure.get(name)
        if isinstance(value, str):
            values.append(value)

    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return tuple(values)
    for name in ("variant", "v"):
        parameter = signature.parameters.get(name)
        if parameter is not None and isinstance(parameter.default, str):
            values.append(parameter.default)
    return tuple(values)


def _audit_registries(
    module: object,
    module_name: str,
    arch_ids: list[str],
) -> tuple[int, int, int, list[str]]:
    registries = [
        (name, value)
        for name, value in vars(module).items()
        if name.endswith("_REGISTRY") and isinstance(value, Mapping)
    ]
    errors: list[str] = []
    builder_count = sum(len(registry) for _, registry in registries)
    if not registries:
        errors.append(f"{module_name}: no *_REGISTRY mapping found")
    if builder_count != len(arch_ids):
        errors.append(
            f"{module_name}: {builder_count} registered builders do not match "
            f"{len(arch_ids)} listed ids"
        )

    listed_names = {arch_id.split(":", 1)[1] for arch_id in arch_ids if ":" in arch_id}
    registry_names = {str(key) for _, registry in registries for key in registry}
    if listed_names != registry_names:
        errors.append(
            f"{module_name}: listed names and registry keys differ "
            f"(unbuilt={sorted(listed_names - registry_names)[:5]}, "
            f"unlisted={sorted(registry_names - listed_names)[:5]})"
        )

    binding_checked = 0
    for registry_name, registry in registries:
        for raw_key, builder in registry.items():
            key = str(raw_key)
            if not callable(builder):
                errors.append(f"{module_name}.{registry_name}[{key!r}] is not callable")
                continue

            variants = _bound_variants(builder)
            if not variants or getattr(builder, "__name__", "") not in {"_build", "_builder"}:
                continue
            binding_checked += 1
            normalized_key = _normalize_bound_name(key)
            if not any(
                _normalize_bound_name(variant) in normalized_key
                or normalized_key in _normalize_bound_name(variant)
                for variant in variants
            ):
                errors.append(
                    f"{module_name}.{registry_name}[{key!r}] captures unrelated "
                    f"variant(s) {variants!r}; possible late binding"
                )

    return len(registries), builder_count, binding_checked, errors


def audit_zoo_integrity(root: Path | None = None) -> ZooAudit:
    root = (root or repo_root()).resolve()
    _ensure_repo_root_on_path(root)
    modules: list[ModuleAudit] = []
    errors: list[str] = []
    id_owners: dict[str, list[str]] = {}
    namespace_owners: dict[str, set[str]] = {}

    with block_network():
        for path in discover_zoo_paths(root):
            module_name = _module_name(path, root)
            relative_path = path.relative_to(root).as_posix()
            parsed = False
            imported = False
            arch_ids: list[str] = []
            namespaces: tuple[str, ...] = ()
            registry_count = 0
            builder_count = 0
            binding_checked = 0

            try:
                tree = ast.parse(_read_python(path), filename=str(path))
            except (LookupError, OSError, SyntaxError, UnicodeError) as exc:
                errors.append(f"{module_name}: cannot parse {relative_path}: {exc}")
            else:
                parsed = True
                functions = _top_level_functions(tree)
                if "list_local_arches" not in functions:
                    errors.append(f"{module_name}: missing list_local_arches()")
                if not {"build_local_model", "build_local_strategy"}.intersection(functions):
                    errors.append(
                        f"{module_name}: missing build_local_model()/build_local_strategy()"
                    )

            if parsed:
                try:
                    module = importlib.import_module(module_name)
                    imported = True
                    first = module.list_local_arches()
                    second = module.list_local_arches()
                except Exception as exc:
                    errors.append(f"{module_name}: import/list failed: {type(exc).__name__}: {exc}")
                else:
                    if not isinstance(first, list) or not all(
                        isinstance(arch_id, str) for arch_id in first
                    ):
                        errors.append(f"{module_name}: list_local_arches() must return list[str]")
                    else:
                        arch_ids = first
                        if first != second:
                            errors.append(f"{module_name}: list_local_arches() is not stable")
                        if first != sorted(first):
                            errors.append(f"{module_name}: list_local_arches() is not sorted")
                        if len(first) != len(set(first)):
                            errors.append(f"{module_name}: list_local_arches() has duplicate ids")
                        invalid_ids = [
                            arch_id for arch_id in first if not ARCH_ID_PATTERN.fullmatch(arch_id)
                        ]
                        if invalid_ids:
                            errors.append(
                                f"{module_name}: invalid namespaced ids {invalid_ids[:5]!r}"
                            )

                        namespaces = tuple(sorted({arch_id.split(":", 1)[0] for arch_id in first}))
                        for arch_id in first:
                            id_owners.setdefault(arch_id, []).append(module_name)
                        for namespace in namespaces:
                            namespace_owners.setdefault(namespace, set()).add(module_name)

                        (
                            registry_count,
                            builder_count,
                            binding_checked,
                            registry_errors,
                        ) = _audit_registries(module, module_name, first)
                        errors.extend(registry_errors)

            modules.append(
                ModuleAudit(
                    module=module_name,
                    path=relative_path,
                    parsed=parsed,
                    imported=imported,
                    id_count=len(arch_ids),
                    namespaces=namespaces,
                    registry_count=registry_count,
                    builder_count=builder_count,
                    binding_checked=binding_checked,
                )
            )

    for arch_id, owners in sorted(id_owners.items()):
        if len(owners) > 1:
            errors.append(f"duplicate global arch id {arch_id!r}: {owners!r}")
    for namespace, owners in sorted(namespace_owners.items()):
        if len(owners) > 1:
            errors.append(
                f"namespace {namespace!r} is owned by multiple modules: {sorted(owners)!r}"
            )

    return ZooAudit(modules=tuple(modules), errors=tuple(errors))


def _finite_tensor_count(value: Any) -> int:
    import torch

    if torch.is_tensor(value):
        if value.numel() == 0:
            raise AssertionError("representative output contains an empty tensor")
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise AssertionError("representative output contains a non-finite tensor")
        return 1
    if isinstance(value, Mapping):
        return sum(_finite_tensor_count(child) for child in value.values())
    if isinstance(value, list | tuple):
        return sum(_finite_tensor_count(child) for child in value)
    return 0


def _run_smoke(
    *,
    domain: str,
    module: str,
    arch_id: str,
    operation: str,
    callback,
) -> SmokeAudit:
    started = time.monotonic()
    try:
        output = callback()
        tensor_count = _finite_tensor_count(output)
        if tensor_count == 0:
            raise AssertionError("representative output contains no tensors")
    except Exception as exc:
        return SmokeAudit(
            domain=domain,
            module=module,
            arch_id=arch_id,
            operation=operation,
            tensor_count=0,
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return SmokeAudit(
        domain=domain,
        module=module,
        arch_id=arch_id,
        operation=operation,
        tensor_count=tensor_count,
        elapsed_seconds=time.monotonic() - started,
    )


def run_representative_smokes(root: Path | None = None) -> tuple[SmokeAudit, ...]:
    root = (root or repo_root()).resolve()
    _ensure_repo_root_on_path(root)

    with block_network():
        import torch

        torch.manual_seed(0)

        def federated():
            from dlhub.federated_zoo import build_local_strategy

            strategy = build_local_strategy(
                "dlfed:fedavg_tiny",
                param_dim=8,
                num_clients=2,
                local_steps=1,
                width_mult=0.5,
            )
            return strategy.simulate_round(seed=0)

        def generative():
            from dlhub.generative.gan_zoo import build_local_model

            model = build_local_model(
                "gan:dcgan_tiny",
                in_channels=3,
                image_size=32,
                latent_dim=32,
                num_classes=4,
                width_mult=0.5,
                dropout=0.0,
            ).eval()
            with torch.no_grad():
                return model.forward(batch_size=1)

        def multimodal():
            from dlhub.multimodal.vlm_zoo import build_local_model

            model = build_local_model(
                "vlm:clip_tiny",
                image_size=32,
                vocab_size=64,
                seq_len=8,
                embed_dim=32,
                num_classes=4,
                width_mult=0.5,
                dropout=0.0,
            ).eval()
            with torch.no_grad():
                return model.forward(batch_size=1)

        def nlp():
            from dlhub.nlp.local_zoo import build_local_model

            model = build_local_model(
                "nl:mean_pool",
                vocab_size=64,
                pad_id=0,
                max_length=8,
                num_classes=4,
                width_mult=0.5,
                dropout=0.0,
            ).eval()
            inputs = {
                "input_ids": torch.zeros(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8),
            }
            with torch.no_grad():
                return model(inputs)

        def pointcloud():
            from dlhub.pointcloud.local_zoo import build_local_model

            model = build_local_model(
                "pc:pointnet",
                in_channels=3,
                num_classes=4,
                num_points=16,
                width_mult=0.5,
            ).eval()
            with torch.no_grad():
                return model(torch.zeros(1, 16, 3))

        def vision():
            from dlhub.vision.local_zoo import build_local_model

            model = build_local_model(
                "dl:lenet5",
                in_channels=1,
                num_classes=4,
                image_size=64,
                width_mult=0.5,
            ).eval()
            with torch.no_grad():
                return model(torch.zeros(1, 1, 64, 64))

        cases = (
            ("federated", "dlhub.federated_zoo", "dlfed:fedavg_tiny", "simulate_round", federated),
            ("generative", "dlhub.generative.gan_zoo", "gan:dcgan_tiny", "forward", generative),
            ("multimodal", "dlhub.multimodal.vlm_zoo", "vlm:clip_tiny", "forward", multimodal),
            ("nlp", "dlhub.nlp.local_zoo", "nl:mean_pool", "forward", nlp),
            ("pointcloud", "dlhub.pointcloud.local_zoo", "pc:pointnet", "forward", pointcloud),
            ("vision", "dlhub.vision.local_zoo", "dl:lenet5", "forward", vision),
        )
        return tuple(
            _run_smoke(
                domain=domain,
                module=module,
                arch_id=arch_id,
                operation=operation,
                callback=callback,
            )
            for domain, module, arch_id, operation, callback in cases
        )


def _format_summary(audit: ZooAudit, smokes: tuple[SmokeAudit, ...]) -> str:
    summary = audit.summary
    passed_smokes = sum(smoke.error is None for smoke in smokes)
    lines = [
        f"zoo integrity: {summary['modules_imported']}/{summary['modules_discovered']} modules imported",
        f"- namespaces: {summary['namespaces']} single-owner",
        f"- architecture ids: {summary['ids']} listed, {summary['builders']} builders",
        f"- dynamic builder bindings checked: {summary['bindings_checked']}",
        "- imports/lists/builds: network blocked",
    ]
    if smokes:
        lines.append(f"- representative runtime: {passed_smokes}/{len(smokes)} domains")
        lines.extend(
            f"  - {smoke.domain}: {smoke.arch_id} {smoke.operation} "
            f"({smoke.tensor_count} tensors, {smoke.elapsed_seconds:.3f}s)"
            for smoke in smokes
            if smoke.error is None
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit every local Model Zoo registry deterministically and offline."
    )
    parser.add_argument("--check", action="store_true", help="Exit non-zero on integrity errors.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build and execute one minimal representative per domain.",
    )
    parser.add_argument("--json", action="store_true", help="Print one machine-readable document.")
    args = parser.parse_args(argv)
    if not args.check and not args.smoke and not args.json:
        parser.error("choose --check, --smoke, or --json")

    root = repo_root()
    audit = audit_zoo_integrity(root)
    smokes = run_representative_smokes(root) if args.smoke else ()
    smoke_errors = [
        f"{smoke.domain} representative {smoke.arch_id}: {smoke.error}"
        for smoke in smokes
        if smoke.error is not None
    ]
    errors = [*audit.errors, *smoke_errors]

    if args.json:
        print(
            json.dumps(
                {
                    "summary": {
                        **audit.summary,
                        "smokes_passed": sum(smoke.error is None for smoke in smokes),
                        "smokes_total": len(smokes),
                        "ok": not errors,
                    },
                    "modules": [asdict(module) for module in audit.modules],
                    "smokes": [asdict(smoke) for smoke in smokes],
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(_format_summary(audit, smokes))
        if errors:
            print(f"zoo integrity: FAILED ({len(errors)} errors)")
            for error in errors:
                print(f"- {error}")
        elif args.check or args.smoke:
            print("zoo integrity: OK")

    return 1 if errors and (args.check or args.smoke) else 0


if __name__ == "__main__":
    raise SystemExit(main())
