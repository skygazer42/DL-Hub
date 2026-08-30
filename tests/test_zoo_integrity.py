from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from scripts.zoo_integrity import audit_zoo_integrity


def test_all_zoo_modules_have_one_deterministic_global_inventory() -> None:
    audit = audit_zoo_integrity()
    summary = audit.summary

    assert audit.errors == ()
    assert summary == {
        "modules_discovered": 123,
        "modules_parsed": 123,
        "modules_imported": 123,
        "namespaces": 124,
        "ids": 8611,
        "registries": 124,
        "builders": 8611,
        "bindings_checked": 8233,
        "network_blocked": True,
        "ok": True,
    }


def test_lazy_family_registry_binds_each_family_and_variant(monkeypatch) -> None:
    from dlhub.zoo_registry import make_lazy_family_registry

    calls: list[tuple[str, str, str]] = []

    def make_builder(family: str):
        def build(*, marker: str, variant: str):
            calls.append((family, marker, variant))
            return family, variant

        return build

    modules = {
        "example.alpha": SimpleNamespace(build_alpha=make_builder("alpha")),
        "example.beta": SimpleNamespace(build_beta=make_builder("beta")),
    }
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda module_name: modules[module_name],
    )

    registry = make_lazy_family_registry(
        ("alpha", "beta"),
        ("tiny", "small"),
        module_template="example.{family}",
        builder_template="build_{family}",
        kwargs_factory=lambda marker, variant: {"marker": marker, "variant": variant},
    )
    outputs = {key: builder("sentinel") for key, builder in registry.items()}

    assert outputs == {
        "alpha_tiny": ("alpha", "alpha_tiny"),
        "alpha_small": ("alpha", "alpha_small"),
        "beta_tiny": ("beta", "beta_tiny"),
        "beta_small": ("beta", "beta_small"),
    }
    assert calls == [
        ("alpha", "sentinel", "alpha_tiny"),
        ("alpha", "sentinel", "alpha_small"),
        ("beta", "sentinel", "beta_tiny"),
        ("beta", "sentinel", "beta_small"),
    ]


def test_zoo_integrity_cli_runs_all_domain_representatives() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    proc = subprocess.run(
        [sys.executable, "scripts/zoo_integrity.py", "--check", "--smoke", "--json"],
        cwd=str(os.fspath(os.path.dirname(os.path.dirname(__file__)))),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["errors"] == []
    assert payload["summary"]["modules_imported"] == 123
    assert payload["summary"]["ids"] == 8611
    assert payload["summary"]["smokes_passed"] == 6
    assert payload["summary"]["smokes_total"] == 6
    assert {smoke["domain"] for smoke in payload["smokes"]} == {
        "federated",
        "generative",
        "multimodal",
        "nlp",
        "pointcloud",
        "vision",
    }
    assert all(smoke["error"] is None for smoke in payload["smokes"])
    assert all(smoke["tensor_count"] > 0 for smoke in payload["smokes"])
