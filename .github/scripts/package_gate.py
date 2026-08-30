#!/usr/bin/env python3
"""Clean, validate, and smoke-test DL-Hub distribution artifacts."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import os
from pathlib import Path, PurePosixPath
import runpy
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import venv
from zipfile import ZipFile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATTERNS = ("dlhub-*.whl", "dlhub-*.tar.gz")
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
PRIVATE_KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
)
FORBIDDEN_PATH_PARTS = {".git", ".github", "resources", "scripts", "tracks"}
ALLOWED_SDIST_TOP_LEVEL = {
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "dlhub",
    "dlhub.egg-info",
    "pyproject.toml",
    "setup.cfg",
    "tests",
}


def _source_project() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_member_name(name: str) -> PurePosixPath:
    raw_name = name[:-1] if name.endswith("/") else name
    raw_parts = raw_name.split("/")
    path = PurePosixPath(raw_name)
    if (
        not raw_name
        or path.is_absolute()
        or "\\" in raw_name
        or ":" in raw_name
        or "\x00" in raw_name
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return path


def _is_credential_like(path: PurePosixPath) -> bool:
    filename = path.name.lower()
    return (
        filename == ".env"
        or filename.startswith(".env.")
        or filename
        in {
            ".npmrc",
            ".pypirc",
            "credentials",
            "credentials.json",
            "id_dsa",
            "id_ed25519",
            "id_rsa",
        }
        or filename.endswith((".kdbx", ".key", ".p12", ".pem", ".pfx"))
    )


def _validate_archive_limits(path: Path, *, members: int, expanded_bytes: int) -> None:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive {path.name} is too large: {path.stat().st_size} bytes "
            f"(maximum {MAX_ARCHIVE_BYTES})"
        )
    if members > MAX_ARCHIVE_MEMBERS:
        raise RuntimeError(
            f"archive {path.name} contains too many members: {members} "
            f"(maximum {MAX_ARCHIVE_MEMBERS})"
        )
    if expanded_bytes > MAX_EXPANDED_BYTES:
        raise RuntimeError(
            f"archive {path.name} expands to {expanded_bytes} bytes "
            f"(maximum {MAX_EXPANDED_BYTES})"
        )


def _validate_private_key_content(name: str, data: bytes) -> None:
    if any(marker in data for marker in PRIVATE_KEY_MARKERS):
        raise RuntimeError(f"archive member contains private-key material: {name}")


def _metadata_body(raw_metadata: bytes) -> bytes:
    try:
        return raw_metadata.split(b"\n\n", 1)[1]
    except IndexError as exc:
        raise RuntimeError("distribution metadata has no long description") from exc


def _validate_metadata(metadata, *, label: str) -> None:
    project = _source_project()
    expected = {
        "Name": project["name"],
        "Version": _source_version(),
        "Requires-Python": project["requires-python"],
        "License-Expression": project["license"],
        "Description-Content-Type": "text/markdown",
    }
    for header, value in expected.items():
        if str(metadata.get(header, "")) != value:
            raise RuntimeError(
                f"{label} metadata {header} is {metadata.get(header)!r}, expected {value!r}"
            )
    if metadata.get_all("License-File", []) != ["LICENSE"]:
        raise RuntimeError(
            f"{label} metadata must declare LICENSE, found {metadata.get_all('License-File', [])}"
        )
    base_requirements = [
        requirement
        for requirement in metadata.get_all("Requires-Dist", [])
        if "extra ==" not in requirement
    ]
    if base_requirements != project["dependencies"]:
        raise RuntimeError(
            f"{label} base requirements {base_requirements} do not match "
            f"pyproject.toml {project['dependencies']}"
        )


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _artifacts(dist_dir: Path) -> tuple[list[Path], list[Path]]:
    wheels = sorted(dist_dir.glob("dlhub-*.whl"))
    sdists = sorted(dist_dir.glob("dlhub-*.tar.gz"))
    return wheels, sdists


def clean_artifacts(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted({path for pattern in ARTIFACT_PATTERNS for path in dist_dir.glob(pattern)})
    for path in stale:
        path.unlink()
        print(f"removed stale artifact: {path}")
    if not stale:
        print(f"no stale dlhub artifacts in {dist_dir}")


def _require_fresh_artifacts(dist_dir: Path) -> tuple[Path, Path]:
    wheels, sdists = _artifacts(dist_dir)
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"expected exactly one wheel and one sdist in {dist_dir}, "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    return wheels[0], sdists[0]


def _source_version() -> str:
    about = runpy.run_path(str(REPO_ROOT / "dlhub" / "__about__.py"))
    return str(about["__version__"])


def _audit_wheel(wheel: Path):
    with ZipFile(wheel) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"wheel CRC check failed for member: {corrupt}")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError(f"wheel contains duplicate member names: {wheel}")
        expanded_bytes = sum(info.file_size for info in infos)
        _validate_archive_limits(wheel, members=len(infos), expanded_bytes=expanded_bytes)

        paths: list[PurePosixPath] = []
        for info in infos:
            path = _validate_archive_member_name(info.filename)
            paths.append(path)
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise RuntimeError(f"wheel contains a link or special member: {info.filename}")
            if info.flag_bits & 0x1:
                raise RuntimeError(f"wheel contains an encrypted member: {info.filename}")
            if _is_credential_like(path):
                raise RuntimeError(f"wheel contains a credential-like file: {info.filename}")
            if any(part.lower() in FORBIDDEN_PATH_PARTS for part in path.parts):
                raise RuntimeError(f"wheel contains a repository-only path: {info.filename}")
            if "tests" in {part.lower() for part in path.parts}:
                raise RuntimeError(f"wheel unexpectedly contains tests: {info.filename}")
            if path.parts[0] == "dlhub" and not info.is_dir() and path.suffix != ".py":
                raise RuntimeError(
                    f"wheel unexpectedly contains a non-Python package resource: {info.filename}"
                )
            if not info.is_dir():
                _validate_private_key_content(info.filename, archive.read(info))

        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"expected one METADATA file in {wheel}, found {len(metadata_files)}"
            )
        metadata_raw = archive.read(metadata_files[0])
        metadata = BytesParser(policy=policy.default).parsebytes(metadata_raw)
        _validate_metadata(metadata, label="wheel")
        dist_info = PurePosixPath(metadata_files[0]).parts[0]
        required = {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/licenses/LICENSE",
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"wheel is missing required members: {sorted(missing)}")
        if archive.read(f"{dist_info}/licenses/LICENSE") != (REPO_ROOT / "LICENSE").read_bytes():
            raise RuntimeError("wheel LICENSE does not match the repository LICENSE")
        if _metadata_body(metadata_raw) != (REPO_ROOT / "README.md").read_bytes():
            raise RuntimeError("wheel long description does not match README.md")

        top_levels = {path.parts[0] for path in paths}
        if top_levels != {"dlhub", dist_info}:
            raise RuntimeError(f"unexpected wheel top-level members: {sorted(top_levels)}")

    print(
        f"artifact audit: {wheel.name}: {len(infos)} members, "
        f"{wheel.stat().st_size} compressed bytes, {expanded_bytes} expanded bytes, "
        f"sha256={_sha256(wheel)}"
    )
    return metadata, top_levels


def _audit_sdist(sdist: Path):
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(f"sdist contains duplicate member names: {sdist}")
        expanded_bytes = sum(member.size for member in members if member.isfile())
        _validate_archive_limits(sdist, members=len(members), expanded_bytes=expanded_bytes)

        paths: list[PurePosixPath] = []
        for member in members:
            path = _validate_archive_member_name(member.name)
            paths.append(path)
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"sdist contains a link or special member: {member.name}")
            if _is_credential_like(path):
                raise RuntimeError(f"sdist contains a credential-like file: {member.name}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot read sdist member: {member.name}")
                _validate_private_key_content(member.name, stream.read())

        roots = {path.parts[0] for path in paths}
        if len(roots) != 1:
            raise RuntimeError(f"sdist must have exactly one root directory, found {sorted(roots)}")
        root = roots.pop()
        expected_root = f"dlhub-{_source_version()}"
        if root != expected_root:
            raise RuntimeError(f"sdist root is {root!r}, expected {expected_root!r}")

        relative_paths = [path.parts[1:] for path in paths if len(path.parts) > 1]
        top_levels = {parts[0] for parts in relative_paths}
        unexpected = top_levels - ALLOWED_SDIST_TOP_LEVEL
        if unexpected:
            raise RuntimeError(f"unexpected sdist top-level members: {sorted(unexpected)}")
        for parts in relative_paths:
            if any(part.lower() in FORBIDDEN_PATH_PARTS for part in parts):
                raise RuntimeError(
                    "sdist contains a repository-only path: " + "/".join((root, *parts))
                )
        for member in members:
            path = PurePosixPath(member.name)
            if (
                member.isfile()
                and len(path.parts) > 2
                and path.parts[1] == "dlhub"
                and path.suffix != ".py"
            ):
                raise RuntimeError(
                    f"sdist unexpectedly contains a non-Python package resource: {member.name}"
                )

        required = {
            f"{root}/{name}" for name in ("LICENSE", "PKG-INFO", "README.md", "pyproject.toml")
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"sdist is missing required members: {sorted(missing)}")
        for filename in ("LICENSE", "README.md", "pyproject.toml"):
            stream = archive.extractfile(f"{root}/{filename}")
            if stream is None or stream.read() != (REPO_ROOT / filename).read_bytes():
                raise RuntimeError(f"sdist {filename} does not match the repository copy")

        pkg_info_stream = archive.extractfile(f"{root}/PKG-INFO")
        if pkg_info_stream is None:
            raise RuntimeError("cannot read sdist PKG-INFO")
        metadata_raw = pkg_info_stream.read()
        metadata = BytesParser(policy=policy.default).parsebytes(metadata_raw)
        _validate_metadata(metadata, label="sdist")
        if _metadata_body(metadata_raw) != (REPO_ROOT / "README.md").read_bytes():
            raise RuntimeError("sdist long description does not match README.md")
        test_files = sum(
            member.isfile()
            and len(PurePosixPath(member.name).parts) > 1
            and PurePosixPath(member.name).parts[1] == "tests"
            for member in members
        )

    print(
        f"artifact audit: {sdist.name}: {len(members)} members ({test_files} test files), "
        f"{sdist.stat().st_size} compressed bytes, {expanded_bytes} expanded bytes, "
        f"sha256={_sha256(sdist)}"
    )
    return metadata


def audit_artifacts(wheel: Path, sdist: Path) -> None:
    expected_version = _source_version()
    if not wheel.name.startswith(f"dlhub-{expected_version}-"):
        raise RuntimeError(f"wheel filename does not match source version: {wheel.name}")
    if sdist.name != f"dlhub-{expected_version}.tar.gz":
        raise RuntimeError(f"sdist filename does not match source version: {sdist.name}")
    _audit_wheel(wheel)
    _audit_sdist(sdist)
    print("artifact content audit: OK")


def check_artifacts(dist_dir: Path) -> None:
    wheel, sdist = _require_fresh_artifacts(dist_dir)
    audit_artifacts(wheel, sdist)
    _run([sys.executable, "-m", "twine", "check", str(wheel), str(sdist)])


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _isolated_env(temporary_path: Path) -> dict[str, str]:
    isolated_env = os.environ.copy()
    isolated_env.pop("PYTHONHOME", None)
    isolated_env.pop("PYTHONPATH", None)
    isolated_env["PYTHONNOUSERSITE"] = "1"
    isolated_env["PIP_CACHE_DIR"] = str(temporary_path / "pip-cache")
    return isolated_env


def _installed_validation(expected_version: str) -> str:
    return f"""
import importlib.metadata as metadata
import importlib.util
import json
import re

import dlhub
import numpy as np
from dlhub.metrics import accuracy_numpy

normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
distribution_version = metadata.version("dlhub")
assert dlhub.__version__ == distribution_version == {expected_version!r}
assert accuracy_numpy(np.array([0, 1, 1]), np.array([0, 1, 1])) == 1.0
assert importlib.util.find_spec("torch") is None
assert importlib.util.find_spec("tracks") is None
installed = {{normalize(dist.metadata["Name"]) for dist in metadata.distributions()}}
allowed = {{"dlhub", "numpy", "pip", "setuptools", "wheel"}}
assert not installed - allowed, sorted(installed - allowed)
print(json.dumps({{
    "dlhub_version": dlhub.__version__,
    "numpy_version": np.__version__,
    "package_file": dlhub.__file__,
    "tracks_importable": False,
    "torch_importable": False,
}}, sort_keys=True))
"""


def _install_and_validate(wheel: Path, *, prefix: str, label: str) -> None:
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        temporary_path = Path(temporary)
        venv_dir = temporary_path / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
        python = _venv_python(venv_dir)
        isolated_env = _isolated_env(temporary_path)

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                str(wheel.resolve()),
            ],
            cwd=temporary_path,
            env=isolated_env,
        )
        _run(
            [str(python), "-c", _installed_validation(_source_version())],
            cwd=temporary_path,
            env=isolated_env,
        )

    if temporary_path is None or temporary_path.exists():
        raise RuntimeError(f"temporary {label} environment was not cleaned up")
    print(f"{label}: OK (temporary environment cleaned)")


def smoke_wheel(wheel: Path) -> None:
    _install_and_validate(
        wheel,
        prefix="dlhub-wheel-smoke-",
        label="wheel install smoke",
    )


def smoke_sdist(sdist: Path) -> None:
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="dlhub-sdist-build-") as temporary:
        temporary_path = Path(temporary)
        build_venv = temporary_path / "build-venv"
        wheelhouse = temporary_path / "wheelhouse"
        wheelhouse.mkdir()
        venv.EnvBuilder(with_pip=True, clear=True).create(build_venv)
        python = _venv_python(build_venv)
        isolated_env = _isolated_env(temporary_path)
        _run(
            [
                str(python),
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-input",
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
                str(sdist.resolve()),
            ],
            cwd=temporary_path,
            env=isolated_env,
        )
        rebuilt = sorted(wheelhouse.glob("dlhub-*.whl"))
        if len(rebuilt) != 1:
            raise RuntimeError(f"expected one wheel rebuilt from sdist, found {len(rebuilt)}")
        _audit_wheel(rebuilt[0])
        _install_and_validate(
            rebuilt[0],
            prefix="dlhub-sdist-install-",
            label="sdist rebuild/install smoke",
        )

    if temporary_path is None or temporary_path.exists():
        raise RuntimeError("temporary sdist build environment was not cleaned up")


def smoke_artifacts(dist_dir: Path) -> None:
    wheel, sdist = _require_fresh_artifacts(dist_dir)
    audit_artifacts(wheel, sdist)
    smoke_wheel(wheel)
    smoke_sdist(sdist)
    print("package smoke: OK (wheel and sdist validated)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("clean", "check", "smoke"))
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist")
    args = parser.parse_args()
    dist_dir = args.dist_dir.resolve()

    if args.command == "clean":
        clean_artifacts(dist_dir)
    elif args.command == "check":
        check_artifacts(dist_dir)
    else:
        smoke_artifacts(dist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
