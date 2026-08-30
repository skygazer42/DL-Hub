import re
from pathlib import Path
import runpy

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_GATE = runpy.run_path(str(REPO_ROOT / ".github/scripts/package_gate.py"))
REQUIREMENT_FORWARDERS = {
    "requirements.txt": "-e .",
    "requirements-dev.txt": "-e .[dev]",
    "requirements-docs.txt": "-e .[docs]",
    "requirements-vision.txt": "-e .[vision]",
}


def _config() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _active_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_track_dependency_extras_are_declared_in_one_source_of_truth() -> None:
    config = _config()
    extras = config["project"]["optional-dependencies"]

    tracks = {
        "foundations",
        "vision",
        "nlp",
        "gnn",
        "pointcloud",
        "generative",
        "llm",
        "multimodal",
    }
    assert tracks <= extras.keys()
    assert all(
        any(requirement.startswith("torch") for requirement in extras[track]) for track in tracks
    )
    runtime_requirements = set().union(*(set(extras[name]) for name in tracks | {"torch"}))
    assert set(extras["all"]) == runtime_requirements
    assert all(
        "torchaudio" not in requirement
        for requirements in extras.values()
        for requirement in requirements
    )


def test_requirement_files_are_exact_compatibility_forwarders() -> None:
    actual_files = {path.name for path in REPO_ROOT.glob("requirements*.txt")}
    assert actual_files == REQUIREMENT_FORWARDERS.keys()

    for filename, target in REQUIREMENT_FORWARDERS.items():
        assert _active_lines(REPO_ROOT / filename) == [target]


def test_minimum_python_and_distribution_boundary_are_explicit() -> None:
    config = _config()

    assert config["project"]["requires-python"] == ">=3.10"
    assert config["project"]["dependencies"] == ["numpy>=1.24"]
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["dlhub*"]
    assert config["tool"]["black"]["target-version"] == ["py310"]
    assert config["tool"]["ruff"]["target-version"] == "py310"


def test_ci_python_matrix_and_installation_docs_match_the_package_contract() -> None:
    config = _config()
    minimum = config["project"]["requires-python"].removeprefix(">=")
    workflow = (REPO_ROOT / ".github/workflows/python-ci.yml").read_text(encoding="utf-8")
    matrix = re.findall(
        r'^\s+- python-version: "([0-9.]+)"\n\s+release_gates: (true|false)$',
        workflow,
        flags=re.MULTILINE,
    )

    assert matrix == [(minimum, "true"), ("3.12", "false")]
    assert workflow.count("python-version: ${{ matrix.python-version }}") == 1

    installation = (REPO_ROOT / "docs/getting-started/installation.md").read_text(encoding="utf-8")
    extras = config["project"]["optional-dependencies"]
    assert "`pyproject.toml` 是依赖契约的唯一来源" in installation
    assert all(f'".[{extra}]"' in installation for extra in extras)
    assert f"最低版本是 Python {minimum}" in installation

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    requirements_doc = (REPO_ROOT / "docs/getting-started/requirements.md").read_text(
        encoding="utf-8"
    )
    assert f"Python-{minimum}+" in readme
    assert f"| **Python** | {minimum}+" in requirements_doc

    for relative in ("docs/developer/testing.md", "docs/developer/release.md"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "Python 3.10" in text
        assert "Python 3.12" in text


def test_maintained_install_commands_bind_pip_to_the_selected_python() -> None:
    maintained_docs = (
        "README.md",
        "docs/STYLEGUIDE.md",
        "docs/faq.md",
        "docs/getting-started/installation.md",
        "docs/getting-started/quickstart.md",
        "docs/developer/contributing.md",
        "docs/developer/release.md",
    )
    bare_pip = re.compile(r"(?<!python -m )pip install")

    for relative in maintained_docs:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert bare_pip.search(text) is None, relative
        assert "torchaudio" not in text, relative


def test_readme_distribution_images_use_absolute_urls() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    image_sources = re.findall(r'<img\s+[^>]*src="([^"]+)"', readme)

    assert image_sources
    assert all(source.startswith("https://") for source in image_sources)


def test_package_gate_rejects_unsafe_or_sensitive_archive_members() -> None:
    validate_name = PACKAGE_GATE["_validate_archive_member_name"]
    is_credential_like = PACKAGE_GATE["_is_credential_like"]
    validate_private_key = PACKAGE_GATE["_validate_private_key_content"]

    assert validate_name("dlhub/module.py").parts == ("dlhub", "module.py")
    for unsafe in (
        "/absolute/path",
        "../escape",
        "root/../escape",
        "root//file",
        r"root\windows-path",
        "C:/windows-path",
    ):
        with pytest.raises(RuntimeError, match="unsafe archive member path"):
            validate_name(unsafe)

    for credential in ("root/.env", "root/id_rsa", "root/token.pem"):
        assert is_credential_like(validate_name(credential)), credential

    with pytest.raises(RuntimeError, match="private-key material"):
        # Assemble the fixture at runtime so the sdist security scan does not
        # mistake this test source for leaked key material.
        private_key_fixture = b"-----BEGIN " + b"PRIVATE KEY-----\nsecret"
        validate_private_key("root/config.txt", private_key_fixture)

    assert {"resources", "scripts", "tracks"} <= PACKAGE_GATE["FORBIDDEN_PATH_PARTS"]
