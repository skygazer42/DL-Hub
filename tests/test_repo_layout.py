
import subprocess
from pathlib import Path


def test_dlhub_data_package_is_not_gitignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "dlhub/data", "dlhub/data/__init__.py", "dlhub/data/splits.py", "dlhub/data/toy.py"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1, (
        "The dlhub.data package must be tracked by git, but ignore rules matched:\n"
        f"{proc.stdout}{proc.stderr}"
    )


def test_tracks_gnn_datasets_package_is_not_gitignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            "git",
            "check-ignore",
            "-v",
            "tracks/gnn/datasets",
            "tracks/gnn/datasets/__init__.py",
            "tracks/gnn/datasets/cora.py",
            "tracks/gnn/datasets/karate.py",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1, (
        "The tracks.gnn.datasets package must be tracked by git, but ignore rules matched:\n"
        f"{proc.stdout}{proc.stderr}"
    )
