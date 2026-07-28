"""Research-stream registry for non-model DL-Hub topics.

Several requested topics are workflow surfaces rather than neural-network
families: paper digests, resource sharing, open-source announcements, surveys,
and dataset/tutorial notices.  This module gives those topics executable,
queryable code artifacts that can be used by docs or future automation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchStream:
    stream_id: str
    title: str
    topics: tuple[str, ...]
    artifact_path: str
    module: str
    primary_action: str


@dataclass(frozen=True)
class StreamDigestItem:
    topic: str
    stream_id: str
    title: str
    artifact_path: str
    module: str
    primary_action: str


_STREAMS: tuple[ResearchStream, ...] = (
    ResearchStream(
        stream_id="notifications",
        title="Notification and update stream",
        topics=("通知",),
        artifact_path="dlhub/research_streams.py",
        module="dlhub.research_streams",
        primary_action="Route repository update notices into a typed stream.",
    ),
    ResearchStream(
        stream_id="papers",
        title="Paper digest stream",
        topics=("论文速递", "论文速递(已开源)", "论文速递(即将开源)"),
        artifact_path="Llms/llm_survey.py",
        module="Llms.llm_survey",
        primary_action="Collect paper-style entries and connect them to runnable code artifacts.",
    ),
    ResearchStream(
        stream_id="papers-open",
        title="Open-source paper digest stream",
        topics=("论文速递(已开源)",),
        artifact_path="Llms/resource_registry.py",
        module="Llms.resource_registry",
        primary_action="Track paper entries with available source-code resources.",
    ),
    ResearchStream(
        stream_id="papers-upcoming",
        title="Upcoming open-source paper digest stream",
        topics=("论文速递(即将开源)",),
        artifact_path="dlhub/research_streams.py",
        module="dlhub.research_streams",
        primary_action="Track announced-but-not-yet-open code entries without pretending code exists.",
    ),
    ResearchStream(
        stream_id="resources",
        title="Resource-sharing stream",
        topics=("资源分享", "开源项目", "电子书"),
        artifact_path="Llms/resource_registry.py",
        module="Llms.resource_registry",
        primary_action="Expose curated resource and project references through code.",
    ),
    ResearchStream(
        stream_id="competitions",
        title="Competition-solution stream",
        topics=("竞赛解决方案",),
        artifact_path="dlhub/research_streams.py",
        module="dlhub.research_streams",
        primary_action="Route competition writeups to reproducible local lessons or zoo artifacts.",
    ),
    ResearchStream(
        stream_id="surveys",
        title="Survey stream",
        topics=("综述", "综述汇总", "Transformer综述"),
        artifact_path="Llms/llm_survey.py",
        module="Llms.llm_survey",
        primary_action="Group survey-style resources and connect them to implementation families.",
    ),
    ResearchStream(
        stream_id="datasets",
        title="Dataset stream",
        topics=("数据集", "新数据集"),
        artifact_path="dlhub/data/compact.py",
        module="dlhub.data.synthetic",
        primary_action="Route dataset topics to reproducible local dataset generators.",
    ),
    ResearchStream(
        stream_id="tutorials",
        title="Tutorial stream",
        topics=("新教程",),
        artifact_path="dlhub/research_streams.py",
        module="dlhub.research_streams",
        primary_action="Route tutorial topics to runnable lesson tracks.",
    ),
)


def list_research_streams() -> list[ResearchStream]:
    return list(_STREAMS)


def find_research_stream(topic: str) -> ResearchStream:
    topic = str(topic)
    for stream in _STREAMS:
        if topic in stream.topics:
            return stream
    raise KeyError(f"Unknown research stream topic: {topic!r}")


def build_stream_digest(topics: list[str] | tuple[str, ...]) -> list[StreamDigestItem]:
    digest: list[StreamDigestItem] = []
    for topic in topics:
        stream = find_research_stream(str(topic))
        digest.append(
            StreamDigestItem(
                topic=str(topic),
                stream_id=stream.stream_id,
                title=stream.title,
                artifact_path=stream.artifact_path,
                module=stream.module,
                primary_action=stream.primary_action,
            )
        )
    return digest


__all__ = [
    "ResearchStream",
    "StreamDigestItem",
    "build_stream_digest",
    "find_research_stream",
    "list_research_streams",
]
