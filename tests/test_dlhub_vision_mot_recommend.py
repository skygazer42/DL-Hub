import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_vision_mot_recommend_profiles_include_core_scenarios() -> None:
    from dlhub.vision.mot._recommend import list_profiles

    keys = {p.key for p in list_profiles()}
    assert "balanced" in keys
    assert "realtime" in keys
    assert "occlusion" in keys
    assert "long_horizon" in keys
    assert "low_compute" in keys


def test_vision_mot_recommend_realtime_returns_online_bias() -> None:
    from dlhub.vision.mot._recommend import recommend_arches
    from dlhub.vision.mot_zoo import list_local_arches

    recs = recommend_arches("realtime", variant="tiny", top_k=8)
    arches = set(list_local_arches())
    assert len(recs) == 8
    assert all(r.arch_id in arches for r in recs)
    assert any(r.group == "online_association" for r in recs[:3])
    assert any(r.family == "sort" for r in recs)
    assert any(r.family == "bytetrack" for r in recs)


def test_vision_mot_zoo_script_recommend_and_profiles() -> None:
    profiles_proc = subprocess.run(
        [sys.executable, "scripts/mot_zoo.py", "--list-profiles"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert profiles_proc.returncode == 0
    assert "Vision MOT recommendation profiles" in profiles_proc.stdout
    assert "realtime" in profiles_proc.stdout

    rec_proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
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
    assert "Vision MOT recommendations" in rec_proc.stdout
    assert "profile=realtime" in rec_proc.stdout
    assert "mot2d:" in rec_proc.stdout


def test_vision_mot_zoo_script_recommend_emits_train_commands() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--variant",
            "tiny",
            "--top-k",
            "3",
            "--emit-train-cmds",
            "--train-device",
            "cpu",
            "--train-epochs",
            "2",
            "--train-max-train-batches",
            "4",
            "--train-max-eval-batches",
            "2",
            "--train-run-prefix",
            "batchA",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Training commands (lesson_14)" in proc.stdout
    assert "python -m tracks.vision.lesson_14_video_mot_basics.train" in proc.stdout
    assert "--arch mot2d:" in proc.stdout
    assert "--device cpu" in proc.stdout
    assert "--epochs 2" in proc.stdout
    assert "--max-train-batches 4" in proc.stdout
    assert "--max-eval-batches 2" in proc.stdout
    assert "--num-samples 128" in proc.stdout
    assert "--batch-size 8" in proc.stdout
    assert "--run-name batchA_01_" in proc.stdout


def test_vision_mot_zoo_script_emit_train_cmds_requires_recommend() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/mot_zoo.py", "--emit-train-cmds"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--emit-train-cmds is only valid with --recommend." in proc.stdout


def test_vision_mot_zoo_script_run_train_cmds_requires_recommend() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/mot_zoo.py", "--run-train-cmds"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--run-train-cmds is only valid with --recommend." in proc.stdout


def test_vision_mot_zoo_script_summary_only_requires_run_train_cmds() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--summary-only",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--summary-only is only valid with --run-train-cmds." in proc.stdout


def test_vision_mot_zoo_script_skip_existing_requires_run_train_cmds() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--skip-existing",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--skip-existing is only valid with --run-train-cmds." in proc.stdout


def test_vision_mot_zoo_script_save_leaderboard_requires_run_train_cmds() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--save-leaderboard",
            "outputs/vision/mot_leaderboard_should_fail.json",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--save-leaderboard is only valid with --run-train-cmds." in proc.stdout


def test_vision_mot_zoo_script_summary_only_with_run_train_cmds(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "summary_only_top1.json"
    artifacts_dir = tmp_path / "summary_only_artifacts"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--variant",
            "tiny",
            "--top-k",
            "1",
            "--run-train-cmds",
            "--summary-only",
            "--train-device",
            "cpu",
            "--train-epochs",
            "1",
            "--train-max-train-batches",
            "1",
            "--train-max-eval-batches",
            "1",
            "--train-num-samples",
            "32",
            "--train-batch-size",
            "4",
            "--train-run-prefix",
            "summary_only_smoke",
            "--save-leaderboard",
            str(out_file),
            "--save-artifacts-dir",
            str(artifacts_dir),
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
    assert f"Saved artifacts to {artifacts_dir}" in proc.stdout
    assert "Running training commands (lesson_14)" not in proc.stdout
    assert "reason:" not in proc.stdout
    assert out_file.is_file()
    assert artifacts_dir.is_dir()


def test_vision_mot_zoo_script_save_artifacts_dir_requires_run_train_cmds() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--save-artifacts-dir",
            "outputs/vision/mot_artifacts_should_fail",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "--save-artifacts-dir is only valid with --run-train-cmds." in proc.stdout


def test_vision_mot_zoo_script_save_artifacts_dir_auto_with_run_train_cmds(
    tmp_path: Path,
) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--variant",
            "tiny",
            "--top-k",
            "1",
            "--run-train-cmds",
            "--train-device",
            "cpu",
            "--train-epochs",
            "1",
            "--train-max-train-batches",
            "1",
            "--train-max-eval-batches",
            "1",
            "--train-num-samples",
            "32",
            "--train-batch-size",
            "4",
            "--train-run-prefix",
            "reco_auto_artifact",
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
    assert (artifacts_dir / "report.md").is_file()
    report = (artifacts_dir / "report.md").read_text(encoding="utf-8")
    assert "# MOT Batch Report" in report
    assert "## Run Results" in report


def test_vision_mot_zoo_script_skip_existing_reuses_outputs(tmp_path: Path) -> None:
    prefix = f"skip_existing_{uuid.uuid4().hex[:8]}"
    common = [
        sys.executable,
        "scripts/mot_zoo.py",
        "--recommend",
        "realtime",
        "--variant",
        "tiny",
        "--top-k",
        "1",
        "--run-train-cmds",
        "--summary-only",
        "--train-device",
        "cpu",
        "--train-epochs",
        "1",
        "--train-max-train-batches",
        "1",
        "--train-max-eval-batches",
        "1",
        "--train-num-samples",
        "32",
        "--train-batch-size",
        "4",
        "--train-run-prefix",
        prefix,
    ]
    first = subprocess.run(
        [*common, "--save-artifacts-dir", str(tmp_path / "first")],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    assert "source=executed" in first.stdout

    second = subprocess.run(
        [*common, "--skip-existing", "--save-artifacts-dir", str(tmp_path / "second")],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0
    assert "source=existing" in second.stdout


def test_vision_mot_zoo_script_recommend_run_train_cmds_top1_smoke(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "mot_leaderboard.json"
    artifacts_dir = tmp_path / "mot_artifacts"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/mot_zoo.py",
            "--recommend",
            "realtime",
            "--variant",
            "tiny",
            "--top-k",
            "1",
            "--run-train-cmds",
            "--train-device",
            "cpu",
            "--train-epochs",
            "1",
            "--train-max-train-batches",
            "1",
            "--train-max-eval-batches",
            "1",
            "--train-num-samples",
            "32",
            "--train-batch-size",
            "4",
            "--train-run-prefix",
            "reco_run_smoke",
            "--save-leaderboard",
            str(out_file),
            "--save-artifacts-dir",
            str(artifacts_dir),
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Running training commands (lesson_14)" in proc.stdout
    assert "Leaderboard (successful runs)" in proc.stdout
    assert "reco_run_smoke_01_" in proc.stdout
    assert "eval_iou=" in proc.stdout
    assert f"Saved leaderboard to {out_file}" in proc.stdout
    assert f"Saved artifacts to {artifacts_dir}" in proc.stdout
    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["profile"] == "realtime"
    assert payload["variant"] == "tiny"
    assert payload["top_k"] == 1
    assert payload["rank_by"] == "iou"
    assert len(payload["run_results"]) == 1
    assert isinstance(payload["leaderboard"], list)
    assert payload["leaderboard"][0]["source"] in {"executed", "existing"}

    assert artifacts_dir.is_dir()
    assert (artifacts_dir / "metadata.json").is_file()
    assert (artifacts_dir / "commands.txt").is_file()
    assert (artifacts_dir / "run_results.json").is_file()
    assert (artifacts_dir / "leaderboard_rows.json").is_file()
    assert (artifacts_dir / "leaderboard.json").is_file()
    assert (artifacts_dir / "leaderboard.csv").is_file()
    assert (artifacts_dir / "logs").is_dir()
    assert any((artifacts_dir / "logs").glob("*.stdout.log"))
    assert any((artifacts_dir / "logs").glob("*.stderr.log"))


def test_vision_mot_save_leaderboard_csv_helper(tmp_path: Path) -> None:
    from scripts.mot_zoo import _save_leaderboard

    out_file = tmp_path / "mot_leaderboard.csv"
    _save_leaderboard(
        path=out_file,
        profile="realtime",
        variant="tiny",
        top_k=2,
        rank_by="iou",
        run_results=[
            {
                "idx": 1,
                "arch_id": "mot2d:sort_tiny",
                "family": "sort",
                "group": "online_association",
                "year": 2016,
                "score": 7.0,
                "run_name": "rt_01_sort",
                "source": "executed",
                "ok": True,
                "returncode": 0,
                "eval_mean_iou": 0.2,
                "eval_presence_acc": 0.8,
                "eval_loss": 1.0,
                "elapsed_sec": 1.23,
            },
            {
                "idx": 2,
                "arch_id": "mot2d:bytetrack_tiny",
                "family": "bytetrack",
                "group": "online_association",
                "year": 2021,
                "score": 7.1,
                "run_name": "rt_02_bytetrack",
                "source": "existing",
                "ok": False,
                "returncode": 1,
                "eval_mean_iou": None,
                "eval_presence_acc": None,
                "eval_loss": None,
                "elapsed_sec": 2.34,
            },
        ],
        leaderboard=[
            {
                "rank": 1,
                "run_name": "rt_01_sort",
                "arch_id": "mot2d:sort_tiny",
                "family": "sort",
                "group": "online_association",
                "year": 2016,
                "score": 7.0,
                "source": "executed",
                "ok": True,
                "returncode": 0,
                "eval_mean_iou": 0.2,
                "eval_presence_acc": 0.8,
                "eval_loss": 1.0,
                "elapsed_sec": 1.23,
            }
        ],
    )
    assert out_file.is_file()
    text = out_file.read_text(encoding="utf-8")
    assert "rank,arch_id,family,group,year,score,run_name,source,ok,returncode,eval_mean_iou,eval_presence_acc,eval_loss,elapsed_sec" in text
    assert "mot2d:sort_tiny" in text
    assert "mot2d:bytetrack_tiny" in text


def test_vision_mot_resolve_artifacts_dir_auto_helper() -> None:
    from scripts.mot_zoo import _resolve_artifacts_dir

    repo_root = _repo_root()
    out = _resolve_artifacts_dir(
        repo_root=repo_root,
        raw="auto",
        profile="realtime",
        variant="tiny",
        top_k=3,
        now=datetime(2026, 3, 14, 16, 0, 1),
    )
    assert out == repo_root / "outputs" / "vision" / "mot_artifacts" / "realtime_tiny_top3_20260314_160001"


def test_vision_mot_write_markdown_report_helper(tmp_path: Path) -> None:
    from scripts.mot_zoo import _write_markdown_report

    out = tmp_path / "report.md"
    _write_markdown_report(
        path=out,
        profile="realtime",
        variant="tiny",
        top_k=2,
        rank_by="iou",
        run_results=[
            {
                "idx": 1,
                "arch_id": "mot2d:sort_tiny",
                "run_name": "rt_01_sort",
                "source": "executed",
                "ok": True,
                "eval_mean_iou": 0.1,
                "eval_presence_acc": 0.8,
                "eval_loss": 1.2,
                "elapsed_sec": 1.23,
            },
            {
                "idx": 2,
                "arch_id": "mot2d:bytetrack_tiny",
                "run_name": "rt_02_bt",
                "source": "existing",
                "ok": False,
                "eval_mean_iou": None,
                "eval_presence_acc": None,
                "eval_loss": None,
                "elapsed_sec": 2.34,
            },
        ],
        leaderboard=[
            {
                "rank": 1,
                "arch_id": "mot2d:sort_tiny",
                "run_name": "rt_01_sort",
                "source": "executed",
                "eval_mean_iou": 0.1,
                "eval_presence_acc": 0.8,
                "eval_loss": 1.2,
                "elapsed_sec": 1.23,
            }
        ],
    )
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "# MOT Batch Report" in text
    assert "## Leaderboard" in text
    assert "mot2d:sort_tiny" in text
    assert "- rank_by: `iou`" in text


def test_vision_mot_rank_successful_runs_helper() -> None:
    from scripts.mot_zoo import _rank_successful_runs

    rows = [
        {"arch_id": "a", "eval_mean_iou": 0.1, "eval_presence_acc": 0.95, "eval_loss": 1.5},
        {"arch_id": "b", "eval_mean_iou": 0.2, "eval_presence_acc": 0.80, "eval_loss": 1.2},
        {"arch_id": "c", "eval_mean_iou": 0.15, "eval_presence_acc": 0.90, "eval_loss": 1.0},
    ]
    by_iou = _rank_successful_runs(rows, rank_by="iou")
    assert [x["arch_id"] for x in by_iou] == ["b", "c", "a"]

    by_acc = _rank_successful_runs(rows, rank_by="acc")
    assert [x["arch_id"] for x in by_acc] == ["a", "c", "b"]

    by_loss = _rank_successful_runs(rows, rank_by="loss")
    assert [x["arch_id"] for x in by_loss] == ["c", "b", "a"]
