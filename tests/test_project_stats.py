"""Guardrails for the auto-generated README/docs statistics.

Three layers:
1. Stats blocks in managed files must match freshly computed values.
2. Known-stale numbers that used to contradict each other must not
   reappear anywhere in the core pages.
3. Stable learning-content stats must not silently shrink. Zoo registration
   growth is governed by the fidelity-audit budget instead of count targets.
"""

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dlhub import project_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stats() -> "project_stats.ProjectStats":
    return project_stats.compute_stats(REPO_ROOT)


def test_stats_blocks_are_in_sync() -> None:
    stale = project_stats.check_files(REPO_ROOT)
    assert not stale, (
        f"Stats blocks out of date in {stale}; run: python scripts/project_stats.py --write"
    )


def test_managed_files_contain_expected_blocks() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    for name in ("hero-badges", "track-overview", "zoo-overview"):
        assert f"<!-- stats:{name} -->" in readme, f"README.md lost stats block {name}"
    assert "<!-- stats:docs-index-stats -->" in docs_index


# Historical wrong values that once contradicted the real numbers. If one of
# these reappears in a core page, someone hand-edited a stat again.
_STALE_PATTERNS = (
    r"736\s*(Architectures|种|个架构)",
    r"813\s*(Architectures|种|个架构)",
    r"393\s*(Test Files|个?\s*pytest|测试文件)",
    r"319\s*lessons",
    r"包含\s*76\s*节课程",
    r"21\s*个(?:\s*Zoo\s*)?子系统",
    r"8000\+",
    r"8545",
)

_CORE_PAGES = (
    "README.md",
    "docs/index.md",
    "docs/tracks",
    "docs/zoo",
    "docs/getting-started",
    "docs/developer/structure.md",
)


def _iter_core_pages():
    for rel in _CORE_PAGES:
        path = REPO_ROOT / rel
        if path.is_dir():
            yield from sorted(path.glob("*.md"))
        else:
            yield path


def test_no_stale_numbers_in_core_pages() -> None:
    offenders = []
    for page in _iter_core_pages():
        text = page.read_text(encoding="utf-8")
        for pattern in _STALE_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{page.relative_to(REPO_ROOT)}:{line}: {match.group(0)!r}")
    assert not offenders, "Stale hand-written stats found:\n" + "\n".join(offenders)


# Audited 2026-07-26. Registration and family counts intentionally do not
# appear here: a larger catalog is not, by itself, a quality improvement.
_NON_REGISTRATION_BASELINE = {
    "test_files": 400,
    "ml_algorithms": 31,
    "vision_backbone_modules": 220,
}

_LESSON_BASELINE = {
    "foundations": 2,
    "vision": 89,
    "nlp": 49,
    "gnn": 11,
    "pointcloud": 36,
    "generative": 51,
    "llm": 43,
    "multimodal": 58,
}


def test_stats_lower_bounds(stats) -> None:
    assert stats.lessons_total >= 339
    for track, minimum in _LESSON_BASELINE.items():
        assert stats.lessons_by_track[track] >= minimum, track
    for name, minimum in _NON_REGISTRATION_BASELINE.items():
        assert getattr(stats, name) >= minimum, name
    assert stats.zoo_modules > 0
    assert stats.total_zoo_ids > 0
    assert stats.baseline_wrappers > 0
    assert stats.baseline_audited_wrappers > 0
    assert stats.baseline_inferred_alias_wrappers > 0
    assert stats.baseline_unreviewed_wrappers == 0
    assert (
        stats.baseline_audited_wrappers
        + stats.baseline_inferred_alias_wrappers
        + stats.baseline_unreviewed_wrappers
        == stats.baseline_wrappers
    )


def test_family_counting_matches_variant_structure(stats) -> None:
    # Each family currently ships tiny/small/base variants.
    assert stats.vlm_zoo_ids == 3 * stats.vlm_families


def test_zoo_overview_uses_registration_not_implementation_counts(stats) -> None:
    rendered = project_stats.render_zoo_overview(stats)

    assert "注册组" in rendered
    assert "架构族" not in rendered
    assert f"{stats.baseline_inferred_alias_wrappers} 源码推断 baseline-alias" in rendered
    assert f"{stats.baseline_unreviewed_wrappers} 未分类 wrapper" in rendered
