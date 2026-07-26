"""Repo-wide statistics used by README/docs stat blocks.

Numbers shown in README.md and docs pages drift when written by hand
(the repo has had 736-vs-791 style contradictions). This module computes
them from the filesystem and the zoo registries, and renders the managed
markdown blocks that ``scripts/project_stats.py --write`` embeds between
``<!-- stats:NAME -->`` / ``<!-- /stats:NAME -->`` markers.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path

TRACKS = (
    "foundations",
    "vision",
    "nlp",
    "gnn",
    "pointcloud",
    "generative",
    "llm",
    "multimodal",
)

_VARIANT_SUFFIXES = ("_tiny", "_small", "_base")

BLOCK_PATTERN = re.compile(
    r"(<!-- stats:(?P<name>[a-z0-9-]+) -->\n)(?P<body>.*?)(<!-- /stats:(?P=name) -->)",
    re.DOTALL,
)

MANAGED_FILES = ("README.md", "docs/index.md")


@dataclass(frozen=True)
class ProjectStats:
    lessons_by_track: dict[str, int] = field(default_factory=dict)
    test_files: int = 0
    ml_algorithms: int = 0
    vision_zoo_ids: int = 0
    vision_backbone_modules: int = 0
    nlp_zoo_ids: int = 0
    pointcloud_zoo_ids: int = 0
    vlm_zoo_ids: int = 0
    vlm_families: int = 0
    gan_families: int = 0
    diffusion_families: int = 0
    federated_families: int = 0
    total_zoo_ids: int = 0
    zoo_modules: int = 0

    @property
    def lessons_total(self) -> int:
        return sum(self.lessons_by_track.values())


def _count_families(arch_ids: list[str]) -> int:
    families = set()
    for arch in arch_ids:
        name = arch.split(":", 1)[-1]
        for suffix in _VARIANT_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        families.add(name)
    return len(families)


def _iter_zoo_modules(repo_root: Path):
    for path in sorted(repo_root.glob("dlhub/**/*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == "local_zoo.py" or path.name.endswith("_zoo.py"):
            parts = path.relative_to(repo_root).with_suffix("").parts
            yield ".".join(parts)


def compute_stats(repo_root: str | Path | None = None) -> ProjectStats:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]

    lessons = {
        track: len(list((root / "tracks" / track).glob("lesson_*"))) for track in TRACKS
    }
    test_files = len(list((root / "tests").rglob("test_*.py")))
    ml_algorithms = len(
        [p for p in (root / "ml_algorithms" / "python").glob("*.py") if p.name != "__init__.py"]
    )

    def arches(module: str) -> list[str]:
        mod = importlib.import_module(module)
        return list(mod.list_local_arches())

    vision_ids = arches("dlhub.vision.local_zoo")
    nlp_ids = arches("dlhub.nlp.local_zoo")
    pointcloud_ids = arches("dlhub.pointcloud.local_zoo")
    vlm_ids = arches("dlhub.multimodal.vlm_zoo")
    gan_ids = arches("dlhub.generative.gan_zoo")
    diffusion_ids = arches("dlhub.generative.diffusion_zoo")
    federated_ids = arches("dlhub.federated_zoo")

    from dlhub.vision.backbones.catalog import list_backbone_modules

    total = 0
    zoo_modules = 0
    for module in _iter_zoo_modules(root):
        mod = importlib.import_module(module)
        lister = getattr(mod, "list_local_arches", None)
        if lister is None:
            continue
        zoo_modules += 1
        total += len(list(lister()))

    return ProjectStats(
        lessons_by_track=lessons,
        test_files=test_files,
        ml_algorithms=ml_algorithms,
        vision_zoo_ids=len(vision_ids),
        vision_backbone_modules=len(list(list_backbone_modules())),
        nlp_zoo_ids=len(nlp_ids),
        pointcloud_zoo_ids=len(pointcloud_ids),
        vlm_zoo_ids=len(vlm_ids),
        vlm_families=_count_families(vlm_ids),
        gan_families=_count_families(gan_ids),
        diffusion_families=_count_families(diffusion_ids),
        federated_families=_count_families(federated_ids),
        total_zoo_ids=total,
        zoo_modules=zoo_modules,
    )


_TRACK_LABELS = {
    "foundations": ("⚡ Foundations / 基础", "docs/tracks/foundations.md"),
    "vision": ("👁️ Vision / 视觉", "docs/tracks/vision.md"),
    "nlp": ("📝 NLP / 自然语言处理", "docs/tracks/nlp.md"),
    "gnn": ("🕸️ GNN / 图神经网络", "docs/tracks/gnn.md"),
    "pointcloud": ("☁️ Point Cloud / 点云", "docs/tracks/pointcloud.md"),
    "generative": ("🎨 Generative / 生成模型", "docs/tracks/generative.md"),
    "llm": ("🤖 LLM / 大语言模型", "docs/tracks/llm.md"),
    "multimodal": ("🌐 Multimodal / 多模态", "docs/tracks/multimodal.md"),
}


def render_hero_badges(stats: ProjectStats) -> str:
    return (
        f"**{stats.lessons_total} Lessons** · **{stats.total_zoo_ids} Model Zoo "
        f"Architectures** · **{stats.ml_algorithms} NumPy ML Algorithms** · "
        f"**{stats.test_files} Test Files**\n"
    )


def render_track_overview(stats: ProjectStats) -> str:
    lines = [
        "| Track | Lessons | 文档 |",
        "|---|---:|---|",
    ]
    for track in TRACKS:
        label, doc = _TRACK_LABELS[track]
        lines.append(f"| {label} | {stats.lessons_by_track[track]} | [{doc}]({doc}) |")
    lines.append(f"| **合计** | **{stats.lessons_total}** | [docs/tracks/](docs/tracks/index.md) |")
    return "\n".join(lines) + "\n"


def render_zoo_overview(stats: ProjectStats) -> str:
    rows = [
        ("Vision Zoo", f"{stats.vision_zoo_ids} 架构 ID / {stats.vision_backbone_modules} 模块", "docs/zoo/vision-zoo.md"),
        ("NLP Zoo", f"{stats.nlp_zoo_ids} 架构 ID", "docs/zoo/nlp-zoo.md"),
        ("Point Cloud Zoo", f"{stats.pointcloud_zoo_ids} 架构 ID", "docs/zoo/pointcloud-zoo.md"),
        ("VLM Zoo", f"{stats.vlm_zoo_ids} ID / {stats.vlm_families} 架构族", "docs/zoo/vlm-zoo.md"),
        ("GAN Zoo", f"{stats.gan_families} 架构族", "docs/zoo/generative-zoo.md"),
        ("Diffusion Zoo", f"{stats.diffusion_families} 架构族", "docs/zoo/generative-zoo.md"),
        ("Federated Zoo", f"{stats.federated_families} 联邦策略族", "docs/zoo/federated-zoo.md"),
    ]
    lines = ["| Zoo | 规模 | 文档 |", "|---|---|---|"]
    for name, size, doc in rows:
        lines.append(f"| {name} | {size} | [{doc}]({doc}) |")
    lines.append(
        f"| **全部 {stats.zoo_modules} 个 zoo 模块合计** | **{stats.total_zoo_ids} 架构 ID** | "
        "[docs/zoo/](docs/zoo/index.md) |"
    )
    return "\n".join(lines) + "\n"


def render_docs_index_stats(stats: ProjectStats) -> str:
    cards = (
        (str(stats.lessons_total), "Lessons"),
        (str(len(TRACKS)), "Learning Tracks"),
        (str(stats.total_zoo_ids), "Model Zoo 架构"),
        (str(stats.ml_algorithms), "ML 算法"),
        (str(stats.test_files), "测试文件"),
    )
    parts = ['<div class="stats-grid" markdown>', ""]
    for number, label in cards:
        parts += [
            '<div class="stat-card" markdown>',
            f'<span class="stat-number">{number}</span>',
            f'<span class="stat-label">{label}</span>',
            "</div>",
            "",
        ]
    parts.append("</div>")
    return "\n".join(parts) + "\n"


RENDERERS = {
    "hero-badges": render_hero_badges,
    "track-overview": render_track_overview,
    "zoo-overview": render_zoo_overview,
    "docs-index-stats": render_docs_index_stats,
}


def apply_blocks(text: str, stats: ProjectStats) -> str:
    def _sub(match: re.Match[str]) -> str:
        name = match.group("name")
        renderer = RENDERERS.get(name)
        if renderer is None:
            raise KeyError(f"Unknown stats block: {name!r}")
        return f"{match.group(1)}{renderer(stats)}{match.group(4)}"

    return BLOCK_PATTERN.sub(_sub, text)


def check_files(repo_root: str | Path | None = None) -> list[str]:
    """Return a list of managed files whose stats blocks are out of date."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    stats = compute_stats(root)
    stale = []
    for rel in MANAGED_FILES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if apply_blocks(text, stats) != text:
            stale.append(rel)
    return stale


def write_files(repo_root: str | Path | None = None) -> list[str]:
    """Rewrite stats blocks in managed files; return the files changed."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    stats = compute_stats(root)
    changed = []
    for rel in MANAGED_FILES:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        updated = apply_blocks(text, stats)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(rel)
    return changed
