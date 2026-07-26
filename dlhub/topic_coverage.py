"""Auditable topic-to-code coverage for the broad DL-Hub topic pool.

The user-facing topic list mixes runnable model families, learning tracks,
resource streams, and optional framework integrations.  This module makes that
coverage explicit and machine-checkable: each requested topic maps to at least
one concrete code artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TopicArtifact:
    path: str
    module: str
    kind: str = "code"
    note: str = ""


@dataclass(frozen=True)
class TopicRecord:
    topic: str
    aliases: tuple[str, ...]
    artifacts: tuple[TopicArtifact, ...]
    coverage: str = "implemented"

    @property
    def primary_artifact(self) -> TopicArtifact:
        return self.artifacts[0]


@dataclass(frozen=True)
class CoverageReport:
    requested_count: int
    covered_count: int
    missing_topics: tuple[str, ...]
    duplicate_topics: tuple[str, ...]


class TopicCoverageError(AssertionError):
    pass


REQUESTED_TOPICS: tuple[str, ...] = (
    "通知",
    "目标检测",
    "论文速递",
    "资源分享",
    "开源项目",
    "综述",
    "综述汇总",
    "论文速递(已开源)",
    "论文速递(即将开源)",
    "Mamba",
    "3DGS",
    "AIGC",
    "多模态",
    "多模态学习",
    "Prompt",
    "扩散模型",
    "语义分割",
    "深度估计",
    "超分辨率",
    "CNN",
    "GAN",
    "目标跟踪",
    "全景分割",
    "竞赛解决方案",
    "人脸识别",
    "电子书",
    "数据增广",
    "人脸检测",
    "数据集",
    "NAS",
    "AutoML",
    "图像分割",
    "新教程",
    "SLAM",
    "实例分割",
    "人体姿态估计",
    "元宇宙",
    "ChatGPT",
    "视频目标分割",
    "Re-ID",
    "医学图像分割",
    "新数据集",
    "显著性目标检测",
    "自动驾驶",
    "人群密度估计",
    "NLP",
    "PyTorch",
    "人脸",
    "车道线检测",
    "车道图估计",
    "去雾",
    "TensorFlow",
    "MXNet",
    "TensorRT",
    "Numpy",
    "行人检测",
    "文本检测",
    "OCR",
    "6D姿态估计",
    "Python",
    "姿态估计",
    "边缘检测",
    "场景文本检测",
    "视频实例分割",
    "3D点云",
    "模型压缩",
    "人脸对齐",
    "去噪",
    "强化学习",
    "行为识别",
    "OpenCV",
    "场景文本识别",
    "去雨",
    "机器学习",
    "风格迁移",
    "视频目标检测",
    "去模糊",
    "显著性检测",
    "剪枝",
    "活体检测",
    "人脸关键点检测",
    "3D目标跟踪",
    "视频修复",
    "人脸表情识别",
    "时序动作检测",
    "图像检索",
    "异常检测",
    "车牌识别",
    "行人属性识别",
    "协同分割",
    "人-物交互检测",
    "假脸检测",
    "人脸解析",
    "弱监督",
    "弱监督目标检测",
    "遥感",
    "遥感图像",
    "视频摘要",
    "视频增强",
    "视频理解",
    "3D目标检测",
    "3D语义分割",
    "3D实例分割",
    "3D全景分割",
    "联机手写汉字识别",
    "图像融合",
    "多焦距图像融合",
    "细粒度",
    "细粒度视觉识别",
    "细粒度视觉分类",
    "细粒度图像分类",
    "草图",
    "草图检索",
    "Few-shot",
    "小样本",
    "Transformer",
    "3DTransformer",
    "轻量级Transformer",
    "点云Transformer",
    "Transformer可解释性",
    "医学Transformer",
    "自监督Transformer",
    "Transformer综述",
    "视频Transformer",
    "胶囊网络",
    "交互式分割",
    "交互式图像分割",
    "伪装物体检测",
    "知识蒸馏",
    "单目深度估计",
    "自监督单目深度估计",
    "自监督",
    "自监督学习",
    "无监督",
    "无监督学习",
    "视频稳像",
    "布局生成",
    "图像合成",
    "联邦学习",
    "视频插帧",
    "反光去除",
    "图像匹配",
    "图像编辑",
    "3D人体姿态估计",
    "时序动作定位",
    "视线估计",
    "文本识别",
    "人群计数",
    "轨迹预测",
    "特征匹配",
    "图像拼接",
    "对抗样本",
    "对抗攻击",
    "遥感目标检测",
    "遥感变化检测",
    "跨视图地理定位",
    "地理定位",
)


def _artifact(path: str, module: str, kind: str = "code", note: str = "") -> TopicArtifact:
    return TopicArtifact(path=path, module=module, kind=kind, note=note)


def _vision(topic: str, package: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                f"dlhub/vision/{package}/__init__.py",
                f"dlhub.vision.{package}",
                note="vision direction package",
            ),
        ),
    )


def _vision_zoo(topic: str, zoo: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                f"dlhub/vision/{zoo}_zoo.py",
                f"dlhub.vision.{zoo}_zoo",
                note="vision local zoo",
            ),
        ),
    )


def _pointcloud(topic: str, package: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                f"dlhub/pointcloud/{package}/__init__.py",
                f"dlhub.pointcloud.{package}",
                note="point-cloud direction package",
            ),
        ),
    )


def _pointcloud_zoo(topic: str, zoo: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                f"dlhub/pointcloud/{zoo}_zoo.py",
                f"dlhub.pointcloud.{zoo}_zoo",
                note="point-cloud local zoo",
            ),
        ),
    )


def _multimodal(topic: str, package: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                f"dlhub/multimodal/{package}/__init__.py",
                f"dlhub.multimodal.{package}",
                note="multimodal direction package",
            ),
        ),
    )


def _generative(topic: str, package: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                f"dlhub/generative/{package}/__init__.py",
                f"dlhub.generative.{package}",
                note="generative direction package",
            ),
        ),
    )


def _method(topic: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                "dlhub/method_kits.py",
                "dlhub.method_kits",
                note="cross-cutting runnable method kit",
            ),
        ),
    )


def _framework(topic: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(
                "dlhub/framework_adapters.py",
                "dlhub.framework_adapters",
                note="optional framework probe adapter",
            ),
        ),
    )


def _stream(topic: str, path: str, module: str, *aliases: str) -> TopicRecord:
    return TopicRecord(
        topic=topic,
        aliases=aliases,
        artifacts=(
            _artifact(path, module, note="research/resource stream surface"),
            _artifact("dlhub/research_streams.py", "dlhub.research_streams"),
        ),
    )


TOPIC_COVERAGE: tuple[TopicRecord, ...] = (
    _stream("通知", "dlhub/research_streams.py", "dlhub.research_streams"),
    _vision_zoo("目标检测", "detection", "检测"),
    _stream("论文速递", "Llms/llm_survey.py", "Llms.llm_survey"),
    _stream("资源分享", "Llms/resource_registry.py", "Llms.resource_registry"),
    _stream("开源项目", "Llms/resource_registry.py", "Llms.resource_registry"),
    _stream("综述", "Llms/llm_survey.py", "Llms.llm_survey"),
    _stream("综述汇总", "Llms/llm_survey.py", "Llms.llm_survey"),
    _stream("论文速递(已开源)", "Llms/resource_registry.py", "Llms.resource_registry"),
    _stream("论文速递(即将开源)", "dlhub/research_streams.py", "dlhub.research_streams"),
    TopicRecord(
        "Mamba",
        ("mamba",),
        (
            _artifact(
                "dlhub/vision/backbones/mambavision.py", "dlhub.vision.backbones.mambavision"
            ),
            _artifact("dlhub/vision/backbones/vmamba.py", "dlhub.vision.backbones.vmamba"),
        ),
    ),
    _pointcloud_zoo("3DGS", "gaussian_splatting", "3D Gaussian Splatting"),
    _vision_zoo("AIGC", "image_synthesis"),
    _multimodal("多模态", "vlm"),
    _multimodal("多模态学习", "audio_visual_learning"),
    _multimodal("Prompt", "prompt_learning", "提示学习"),
    _generative("扩散模型", "diffusion"),
    _vision("语义分割", "segmentation"),
    _vision("深度估计", "depth_estimation"),
    _vision_zoo("超分辨率", "super_resolution"),
    TopicRecord(
        "CNN",
        ("卷积神经网络",),
        (_artifact("dlhub/vision/backbones/cnn.py", "dlhub.vision.backbones.cnn"),),
    ),
    _generative("GAN", "gan"),
    _vision_zoo("目标跟踪", "mot"),
    _vision_zoo("全景分割", "panoptic_segmentation"),
    _stream("竞赛解决方案", "dlhub/research_streams.py", "dlhub.research_streams"),
    _vision("人脸识别", "person_search"),
    _stream("电子书", "Llms/resource_registry.py", "Llms.resource_registry"),
    _vision_zoo("数据增广", "data_augmentation"),
    _vision("人脸检测", "face_detection"),
    _stream("数据集", "dlhub/data/toy.py", "dlhub.data.toy"),
    _method("NAS"),
    _method("AutoML"),
    _vision("图像分割", "segmentation"),
    _stream("新教程", "dlhub/research_streams.py", "dlhub.research_streams"),
    _method("SLAM"),
    _vision_zoo("实例分割", "instance_segmentation"),
    _vision("人体姿态估计", "human_pose_estimation"),
    _method("元宇宙"),
    _vision("ChatGPT", "video_question_answering"),
    _vision("视频目标分割", "video_object_segmentation"),
    _vision("Re-ID", "reid"),
    _vision("医学图像分割", "medical_segmentation"),
    _stream("新数据集", "dlhub/data/toy.py", "dlhub.data.toy"),
    _vision("显著性目标检测", "saliency_detection"),
    _vision("自动驾驶", "lane_detection"),
    _vision("人群密度估计", "crowd_counting"),
    TopicRecord(
        "NLP", ("自然语言处理",), (_artifact("dlhub/nlp/local_zoo.py", "dlhub.nlp.local_zoo"),)
    ),
    _framework("PyTorch"),
    _vision("人脸", "face_detection"),
    _vision_zoo("车道线检测", "lane_detection"),
    _vision("车道图估计", "lane_topology_estimation"),
    _vision("去雾", "dehazing"),
    _framework("TensorFlow"),
    _framework("MXNet"),
    _framework("TensorRT"),
    _framework("Numpy", "NumPy"),
    _vision("行人检测", "person_search"),
    _vision("文本检测", "text_detection"),
    _vision("OCR", "ocr"),
    _vision("6D姿态估计", "sixd_pose_estimation"),
    _framework("Python"),
    _vision("姿态估计", "human_pose_estimation"),
    _vision("边缘检测", "edge_detection"),
    _vision("场景文本检测", "scene_text_spotting"),
    _vision("视频实例分割", "video_instance_segmentation"),
    TopicRecord(
        "3D点云",
        ("点云",),
        (_artifact("dlhub/pointcloud/local_zoo.py", "dlhub.pointcloud.local_zoo"),),
    ),
    _method("模型压缩"),
    _vision("人脸对齐", "face_alignment"),
    _vision("去噪", "denoising"),
    _method("强化学习", "RL"),
    _vision_zoo("行为识别", "action_recognition"),
    _framework("OpenCV"),
    _vision("场景文本识别", "scene_text_spotting"),
    _vision_zoo("去雨", "image_deraining"),
    TopicRecord(
        "机器学习",
        ("ML",),
        (_artifact("ml_algorithms/python/__init__.py", "ml_algorithms.python"),),
    ),
    _vision_zoo("风格迁移", "style_transfer"),
    _vision_zoo("视频目标检测", "video_object_detection"),
    _vision("去模糊", "deblurring"),
    _vision("显著性检测", "saliency_detection"),
    _method("剪枝"),
    _vision("活体检测", "face_anti_spoofing"),
    _vision("人脸关键点检测", "face_alignment"),
    _pointcloud_zoo("3D目标跟踪", "tracking3d"),
    _vision("视频修复", "video_restoration"),
    _vision("人脸表情识别", "facial_expression_recognition"),
    _vision("时序动作检测", "temporal_action_localization"),
    _vision("图像检索", "image_retrieval"),
    _vision("异常检测", "anomaly_detection"),
    _vision("车牌识别", "license_plate_recognition"),
    _vision("行人属性识别", "pedestrian_attribute_analysis"),
    _vision_zoo("协同分割", "co_segmentation"),
    _vision("人-物交互检测", "hoi_detection"),
    _vision("假脸检测", "image_forensics"),
    _vision_zoo("人脸解析", "face_parsing"),
    _vision("弱监督", "weakly_supervised_segmentation"),
    _vision("弱监督目标检测", "weakly_supervised_detection"),
    _vision("遥感", "remote_sensing_detection"),
    _vision("遥感图像", "remote_sensing_change_detection"),
    _vision_zoo("视频摘要", "video_summarization"),
    _vision("视频增强", "video_enhancement"),
    _vision("视频理解", "video_understanding"),
    _pointcloud_zoo("3D目标检测", "detection3d"),
    _pointcloud_zoo("3D语义分割", "segmentation3d"),
    _pointcloud_zoo("3D实例分割", "instance_segmentation3d"),
    _pointcloud_zoo("3D全景分割", "instance_segmentation3d"),
    _vision("联机手写汉字识别", "online_handwriting_recognition"),
    _vision("图像融合", "image_fusion"),
    _vision("多焦距图像融合", "multi_focus_fusion"),
    _vision_zoo("细粒度", "fine_grained_recognition"),
    _vision_zoo("细粒度视觉识别", "fine_grained_recognition"),
    _vision_zoo("细粒度视觉分类", "fine_grained_recognition"),
    _vision_zoo("细粒度图像分类", "fine_grained_recognition"),
    _vision("草图", "sketch_retrieval"),
    _vision("草图检索", "sketch_retrieval"),
    _vision("Few-shot", "few_shot_recognition"),
    _vision("小样本", "few_shot_recognition"),
    TopicRecord(
        "Transformer",
        ("transformer",),
        (_artifact("dlhub/nlp/algorithms/transformer.py", "dlhub.nlp.algorithms.transformer"),),
    ),
    _pointcloud("3DTransformer", "segmentation3d"),
    TopicRecord(
        "轻量级Transformer",
        (),
        (
            _artifact(
                "dlhub/vision/backbones/transformers.py", "dlhub.vision.backbones.transformers"
            ),
        ),
    ),
    _pointcloud("点云Transformer", "segmentation3d"),
    TopicRecord(
        "Transformer可解释性",
        (),
        (
            _artifact(
                "dlhub/nlp/algorithms/_transformer_core.py",
                "dlhub.nlp.algorithms._transformer_core",
            ),
        ),
    ),
    _vision("医学Transformer", "medical_segmentation"),
    _pointcloud("自监督Transformer", "selfsupervised"),
    _stream("Transformer综述", "Llms/llm_survey.py", "Llms.llm_survey"),
    _generative("视频Transformer", "video_diffusion"),
    _method("胶囊网络"),
    _vision("交互式分割", "interactive_segmentation"),
    _vision("交互式图像分割", "interactive_segmentation"),
    _vision("伪装物体检测", "camouflaged_object_detection"),
    _method("知识蒸馏"),
    _vision("单目深度估计", "depth_estimation"),
    _vision("自监督单目深度估计", "depth_estimation"),
    _pointcloud("自监督", "selfsupervised"),
    _pointcloud("自监督学习", "selfsupervised"),
    TopicRecord(
        "无监督",
        ("unsupervised",),
        (_artifact("ml_algorithms/python/clustering.py", "ml_algorithms.python.clustering"),),
    ),
    TopicRecord(
        "无监督学习",
        ("unsupervised learning",),
        (_artifact("ml_algorithms/python/clustering.py", "ml_algorithms.python.clustering"),),
    ),
    _vision_zoo("视频稳像", "video_stabilization"),
    _vision_zoo("布局生成", "layout_generation"),
    _vision_zoo("图像合成", "image_synthesis"),
    TopicRecord(
        "联邦学习",
        ("Federated Learning",),
        (_artifact("dlhub/federated_zoo.py", "dlhub.federated_zoo"),),
    ),
    _vision_zoo("视频插帧", "video_frame_interpolation"),
    _vision("反光去除", "reflection_removal"),
    _vision("图像匹配", "image_matching"),
    _vision("图像编辑", "image_editing"),
    _vision("3D人体姿态估计", "pose_estimation_3d"),
    _vision("时序动作定位", "temporal_action_localization"),
    _vision("视线估计", "gaze_estimation"),
    _vision("文本识别", "text_recognition"),
    _vision("人群计数", "crowd_counting"),
    _vision("轨迹预测", "trajectory_prediction"),
    _vision("特征匹配", "feature_matching"),
    _vision("图像拼接", "image_stitching"),
    _vision_zoo("对抗样本", "adversarial_robustness"),
    _vision_zoo("对抗攻击", "adversarial_robustness"),
    _vision("遥感目标检测", "remote_sensing_detection"),
    _vision("遥感变化检测", "remote_sensing_change_detection"),
    _vision("跨视图地理定位", "cross_view_geo_localization"),
    _vision("地理定位", "geo_localization"),
)


def _normalize(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _records_by_key() -> dict[str, TopicRecord]:
    out: dict[str, TopicRecord] = {}
    for record in TOPIC_COVERAGE:
        for key in (record.topic, *record.aliases):
            out[_normalize(key)] = record
    return out


def iter_topic_records() -> tuple[TopicRecord, ...]:
    """Return records in the same order as the requested topic list."""

    by_key = _records_by_key()
    return tuple(
        by_key[_normalize(topic)] for topic in REQUESTED_TOPICS if _normalize(topic) in by_key
    )


def describe_topic(topic: str) -> TopicRecord:
    key = _normalize(topic)
    record = _records_by_key().get(key)
    if record is None:
        raise KeyError(
            f"Unknown topic: {topic!r}. Add it to TOPIC_COVERAGE before claiming coverage."
        )
    return record


def find_topics(query: str) -> list[TopicRecord]:
    q = _normalize(query)
    if not q:
        return []
    found: list[TopicRecord] = []
    seen: set[str] = set()
    for record in TOPIC_COVERAGE:
        haystack = [_normalize(record.topic), *(_normalize(alias) for alias in record.aliases)]
        if any(q in value for value in haystack):
            if record.topic not in seen:
                found.append(record)
                seen.add(record.topic)
    return found


def coverage_report(topics: tuple[str, ...] | list[str] = REQUESTED_TOPICS) -> CoverageReport:
    by_key = _records_by_key()
    missing = tuple(topic for topic in topics if _normalize(topic) not in by_key)
    seen: set[str] = set()
    duplicates: list[str] = []
    for topic in topics:
        key = _normalize(topic)
        if key in seen:
            duplicates.append(topic)
        seen.add(key)
    return CoverageReport(
        requested_count=len(topics),
        covered_count=len(topics) - len(missing),
        missing_topics=missing,
        duplicate_topics=tuple(duplicates),
    )


def validate_topic_coverage(
    topics: tuple[str, ...] | list[str] = REQUESTED_TOPICS,
    *,
    repo_root: str | Path | None = None,
) -> None:
    report = coverage_report(topics)
    if report.missing_topics:
        missing = ", ".join(report.missing_topics)
        raise TopicCoverageError(
            f"Missing topic coverage for: {missing}. Add it to TOPIC_COVERAGE with at least one concrete artifact."
        )

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    root = Path(repo_root)

    missing_artifacts: list[str] = []
    for topic in topics:
        record = describe_topic(topic)
        if not record.artifacts:
            missing_artifacts.append(f"{topic}: no artifacts")
            continue
        for artifact in record.artifacts:
            if not (root / artifact.path).exists():
                missing_artifacts.append(f"{topic}: {artifact.path}")

    if missing_artifacts:
        raise TopicCoverageError(
            "Missing topic artifacts:\n" + "\n".join(sorted(missing_artifacts))
        )


__all__ = [
    "CoverageReport",
    "REQUESTED_TOPICS",
    "TOPIC_COVERAGE",
    "TopicArtifact",
    "TopicCoverageError",
    "TopicRecord",
    "coverage_report",
    "describe_topic",
    "find_topics",
    "iter_topic_records",
    "validate_topic_coverage",
]
