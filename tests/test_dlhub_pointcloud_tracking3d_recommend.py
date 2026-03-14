import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_tracking3d_recommend_profiles_include_core_scenarios() -> None:
    from dlhub.pointcloud.tracking3d._recommend import list_profiles

    keys = {p.key for p in list_profiles()}
    assert "balanced" in keys
    assert "realtime_lidar" in keys
    assert "bev_priority" in keys
    assert "segmentation_first" in keys
    assert "long_horizon" in keys


def test_tracking3d_recommend_bev_priority_returns_bev_bias() -> None:
    from dlhub.pointcloud.tracking3d._recommend import list_profiles, recommend_arches
    from dlhub.pointcloud.tracking3d_zoo import list_local_arches

    bev_profile = next(p for p in list_profiles() if p.key == "bev_priority")
    bev_family_set = set(bev_profile.preferred_families)
    recs = recommend_arches("bev_priority", variant="tiny", top_k=8)
    arches = set(list_local_arches())
    assert len(recs) == 8
    assert all(r.arch_id in arches for r in recs)
    assert any(r.group == "bev_tracking" for r in recs[:3])
    assert all(r.group == "bev_tracking" for r in recs)
    assert all(r.family in bev_family_set for r in recs)


def test_tracking3d_zoo_script_recommend_and_profiles() -> None:
    profiles_proc = subprocess.run(
        [sys.executable, "scripts/tracking3d_zoo.py", "--list-profiles"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert profiles_proc.returncode == 0
    assert "Tracking3D recommendation profiles" in profiles_proc.stdout
    assert "bev_priority" in profiles_proc.stdout

    rec_proc = subprocess.run(
        [
            sys.executable,
            "scripts/tracking3d_zoo.py",
            "--recommend",
            "bev_priority",
            "--variant",
            "tiny",
            "--top-k",
            "5",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert rec_proc.returncode == 0
    assert "Tracking3D recommendations" in rec_proc.stdout
    assert "profile=bev_priority" in rec_proc.stdout
    assert "pctrk3d:" in rec_proc.stdout


def test_tracking3d_zoo_script_emit_smoke_cmds_requires_recommend() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/tracking3d_zoo.py", "--emit-smoke-cmds"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--emit-smoke-cmds is only valid with --recommend." in proc.stdout


def test_tracking3d_zoo_script_run_smoke_cmds_requires_recommend() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/tracking3d_zoo.py", "--run-smoke-cmds"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--run-smoke-cmds is only valid with --recommend." in proc.stdout


def test_tracking3d_zoo_script_recommend_run_smoke_cmds_top1_smoke(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "tracking3d_leaderboard.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/tracking3d_zoo.py",
            "--recommend",
            "bev_priority",
            "--variant",
            "tiny",
            "--top-k",
            "1",
            "--run-smoke-cmds",
            "--summary-only",
            "--batch-size",
            "1",
            "--seq-len",
            "2",
            "--num-points",
            "64",
            "--save-leaderboard",
            str(out_file),
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Leaderboard (successful runs)" in proc.stdout
    assert "source=executed" in proc.stdout
    assert f"Saved leaderboard to {out_file}" in proc.stdout
    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["profile"] == "bev_priority"
    assert payload["variant"] == "tiny"
    assert payload["top_k"] == 1
    assert len(payload["run_results"]) == 1
    assert isinstance(payload["leaderboard"], list)


def test_tracking3d_zoo_script_save_artifacts_dir_requires_run_smoke_cmds() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/tracking3d_zoo.py",
            "--recommend",
            "bev_priority",
            "--save-artifacts-dir",
            "outputs/pointcloud/tracking3d_artifacts_should_fail",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--save-artifacts-dir is only valid with --run-smoke-cmds." in proc.stdout


def test_tracking3d_zoo_script_save_artifacts_dir_auto_with_run_smoke_cmds(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/tracking3d_zoo.py",
            "--recommend",
            "bev_priority",
            "--variant",
            "tiny",
            "--top-k",
            "1",
            "--run-smoke-cmds",
            "--summary-only",
            "--batch-size",
            "1",
            "--seq-len",
            "2",
            "--num-points",
            "64",
            "--save-artifacts-dir",
            "auto",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    marker = "Saved artifacts to "
    assert marker in proc.stdout
    saved = proc.stdout.split(marker, 1)[1].strip().splitlines()[0].strip()
    artifacts_dir = Path(saved)
    assert artifacts_dir.is_dir()
    assert (artifacts_dir / "metadata.json").is_file()
    assert (artifacts_dir / "commands.txt").is_file()
    assert (artifacts_dir / "run_results.json").is_file()
    assert (artifacts_dir / "leaderboard_rows.json").is_file()
    assert (artifacts_dir / "leaderboard.json").is_file()
    assert (artifacts_dir / "leaderboard.csv").is_file()
    assert (artifacts_dir / "report.md").is_file()
    assert (artifacts_dir / "logs").is_dir()
    assert any((artifacts_dir / "logs").glob("*.stdout.log"))
    assert any((artifacts_dir / "logs").glob("*.stderr.log"))


def test_tracking3d_resolve_artifacts_dir_auto_helper() -> None:
    from scripts.tracking3d_zoo import _resolve_artifacts_dir

    repo_root = _repo_root()
    out = _resolve_artifacts_dir(
        repo_root=repo_root,
        raw="auto",
        profile="bev_priority",
        variant="tiny",
        top_k=3,
        now=datetime(2026, 3, 14, 16, 30, 1),
    )
    assert out == repo_root / "outputs" / "pointcloud" / "tracking3d_artifacts" / "bev_priority_tiny_top3_20260314_163001"
