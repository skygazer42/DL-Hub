from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_track_dependency_extras_are_declared_in_one_source_of_truth() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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
    assert all(any(requirement.startswith("torch") for requirement in extras[track]) for track in tracks)
    assert set(extras["all"]) == set(extras["vision"])

    compatibility_files = {
        "requirements.txt": "",
        "requirements-dev.txt": "dev",
        "requirements-docs.txt": "docs",
        "requirements-vision.txt": "vision",
    }
    for filename, extra in compatibility_files.items():
        compatibility = (REPO_ROOT / filename).read_text(encoding="utf-8")
        target = f"-e .[{extra}]" if extra else "-e ."
        assert target in compatibility.splitlines()
