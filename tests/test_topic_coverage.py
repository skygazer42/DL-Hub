from pathlib import Path
import importlib

import pytest

from dlhub.topic_coverage import (
    REQUESTED_TOPICS,
    TopicCoverageError,
    coverage_report,
    describe_topic,
    find_topics,
    iter_topic_records,
    validate_topic_coverage,
)
from dlhub.framework_adapters import list_framework_adapters, probe_framework
from dlhub.method_kits import (
    Pose2D,
    SceneAsset,
    SearchCandidate,
    compose_pose,
    discounted_returns,
    distillation_temperature_loss,
    epsilon_greedy_action,
    make_magnitude_pruning_mask,
    normalize_capsule_routing,
    rank_nas_candidates,
    summarize_scene_assets,
)
from dlhub.research_streams import build_stream_digest, list_research_streams


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_requested_topics_have_concrete_code_artifacts() -> None:
    report = coverage_report()

    assert report.requested_count >= 150
    assert report.covered_count == report.requested_count
    assert not report.missing_topics

    for record in iter_topic_records():
        assert record.topic in REQUESTED_TOPICS
        assert record.artifacts, record.topic
        for artifact in record.artifacts:
            path = REPO_ROOT / artifact.path
            assert path.exists(), f"{record.topic} points at missing artifact {artifact.path}"
            if artifact.module:
                importlib.import_module(artifact.module)
        assert any(
            artifact.path.endswith(".py") for artifact in record.artifacts
        ), f"{record.topic} has no Python artifact"


def test_topic_lookup_normalizes_synonyms_and_chinese_aliases() -> None:
    assert describe_topic("目标检测").primary_artifact.module == "dlhub.vision.detection_zoo"
    assert describe_topic("3DGS").primary_artifact.module == "dlhub.pointcloud.gaussian_splatting_zoo"
    assert describe_topic("论文速递(已开源)").coverage == "implemented"
    assert describe_topic("跨视图地理定位").primary_artifact.module.endswith(
        "cross_view_geo_localization"
    )

    matches = find_topics("Transformer")
    assert {"Transformer", "视频Transformer", "点云Transformer"}.issubset(
        {record.topic for record in matches}
    )


def test_validate_topic_coverage_reports_actionable_failures() -> None:
    validate_topic_coverage()

    with pytest.raises(TopicCoverageError) as excinfo:
        validate_topic_coverage(["不存在的新方向"])

    message = str(excinfo.value)
    assert "不存在的新方向" in message
    assert "Add it to TOPIC_COVERAGE" in message


def test_meta_topics_have_lightweight_runtime_surfaces() -> None:
    streams = {stream.stream_id for stream in list_research_streams()}
    assert {"notifications", "papers-open", "papers-upcoming", "resources", "surveys"}.issubset(
        streams
    )

    digest = build_stream_digest(["论文速递", "资源分享", "综述汇总"])
    assert [item.topic for item in digest] == ["论文速递", "资源分享", "综述汇总"]
    assert all(item.primary_action for item in digest)


def test_framework_topics_use_optional_dependency_adapters() -> None:
    adapters = {adapter.name for adapter in list_framework_adapters()}
    assert {"pytorch", "tensorflow", "mxnet", "tensorrt", "opencv", "numpy", "python"}.issubset(
        adapters
    )

    probe = probe_framework("TensorRT")
    assert probe.name == "tensorrt"
    assert probe.import_name == "tensorrt"
    assert isinstance(probe.available, bool)


def test_cross_cutting_method_topics_have_runnable_kits() -> None:
    result = rank_nas_candidates(
        [
            SearchCandidate("wide", width=64, depth=4, score=0.81),
            SearchCandidate("compact", width=32, depth=3, score=0.81),
            SearchCandidate("weak", width=16, depth=2, score=0.70),
        ]
    )
    assert result.best.name == "compact"

    assert make_magnitude_pruning_mask([0.1, -2.0, 0.4, 0.01], keep_fraction=0.5) == [
        0,
        1,
        1,
        0,
    ]
    assert distillation_temperature_loss([2.0, 4.0], [1.0, 6.0], temperature=2.0) == 0.625

    pose = compose_pose(Pose2D(1.0, 2.0, 0.0), Pose2D(3.0, 4.0, 0.5))
    assert (pose.x, pose.y, pose.theta) == (4.0, 6.0, 0.5)

    scene = summarize_scene_assets(
        [
            SceneAsset("avatar", "mesh", (0.0, 0.0, 0.0), ("human",)),
            SceneAsset("room", "mesh", (1.0, 0.0, 0.0), ("indoor", "layout")),
        ]
    )
    assert scene == {"count": 2, "modalities": {"mesh": 2}, "tag_count": 3}

    routing = normalize_capsule_routing([0.0, 0.0])
    assert routing.couplings == (0.5, 0.5)

    assert discounted_returns([1.0, 1.0, 1.0], gamma=0.5) == [1.75, 1.5, 1.0]
    assert epsilon_greedy_action([0.1, 0.9, 0.2], epsilon=0.0, step=0) == 1
    assert epsilon_greedy_action([0.1, 0.9, 0.2], epsilon=0.5, step=2) == 2
