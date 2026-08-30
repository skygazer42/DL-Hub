from pathlib import Path

import pytest

from dlhub.paths import build_run_paths, get_repo_root


def test_build_run_paths_uses_override_and_normalizes_spaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    paths = build_run_paths(" vision ", "lesson one", " first run ")

    assert paths.run_dir == tmp_path / "vision" / "lesson_one" / "first_run"
    assert paths.checkpoints_dir == paths.run_dir / "checkpoints"
    assert paths.logs_dir == paths.run_dir / "logs"


def test_relative_outputs_override_remains_anchored_to_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", "custom-outputs")

    paths = build_run_paths("vision", "lesson", "run")

    assert paths.run_dir == get_repo_root() / "custom-outputs" / "vision" / "lesson" / "run"


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "",
        "   ",
        ".",
        "..",
        "../escape",
        "nested/run",
        r"nested\run",
        "/absolute",
        r"C:\absolute",
        "line\nbreak",
        "bad:name",
        "bad?name",
        "trailing.",
        "CON",
    ],
)
def test_build_run_paths_rejects_unsafe_or_nonportable_components(unsafe_name: str) -> None:
    with pytest.raises(ValueError):
        build_run_paths("vision", "lesson", unsafe_name)


@pytest.mark.parametrize("field", ["track", "lesson", "run_name"])
def test_build_run_paths_validates_every_component(field: str) -> None:
    values = {"track": "vision", "lesson": "lesson", "run_name": "run"}
    values[field] = "../escape"

    with pytest.raises(ValueError, match=field):
        build_run_paths(**values)


def test_build_run_paths_rejects_non_string_components() -> None:
    with pytest.raises(TypeError, match="track must be a string"):
        build_run_paths(None, "lesson", "run")  # type: ignore[arg-type]
