
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CONVENTION_DIRS = (
    REPO_ROOT / "dlhub/vision/action_recognition",
    REPO_ROOT / "dlhub/vision/detection",
    REPO_ROOT / "dlhub/vision/denoising",
    REPO_ROOT / "dlhub/vision/fine_grained_recognition",
    REPO_ROOT / "dlhub/vision/instance_segmentation",
    REPO_ROOT / "dlhub/vision/panoptic_segmentation",
    REPO_ROOT / "dlhub/vision/segmentation",
    REPO_ROOT / "dlhub/pointcloud/detection3d",
    REPO_ROOT / "dlhub/pointcloud/tracking3d",
    REPO_ROOT / "dlhub/pointcloud/segmentation3d",
    REPO_ROOT / "dlhub/pointcloud/instance_segmentation3d",
)

NLP_FIRST_BATCH = (
    REPO_ROOT / "dlhub/nlp/algorithms/albert.py",
    REPO_ROOT / "dlhub/nlp/algorithms/bert.py",
    REPO_ROOT / "dlhub/nlp/algorithms/performer.py",
)


def _iter_family_modules(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.glob("*.py"))
        if path.name != "__init__.py" and not path.name.startswith("_")
    ]


def _has_variants(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "_VARIANTS" for target in node.targets):
                return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_VARIANTS":
            return True
    return False


def _has_builder(tree: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("build_")
        for node in tree.body
    )


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


def _missing_conventions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing: list[str] = []
    if not _has_variants(tree):
        missing.append("_VARIANTS")
    if not _has_builder(tree):
        missing.append("build_*")
    if not _has_main_guard(tree):
        missing.append("__main__ smoke")
    return missing


@pytest.mark.parametrize("directory", CONVENTION_DIRS, ids=lambda path: path.name)
def test_recent_zoo_family_modules_follow_conventions(directory: Path) -> None:
    failures: list[str] = []
    for path in _iter_family_modules(directory):
        missing = _missing_conventions(path)
        if missing:
            failures.append(f"{path.relative_to(REPO_ROOT)} missing {', '.join(missing)}")

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("path", NLP_FIRST_BATCH, ids=lambda path: path.stem)
def test_selected_nlp_family_modules_follow_conventions(path: Path) -> None:
    missing = _missing_conventions(path)
    assert not missing, f"{path.relative_to(REPO_ROOT)} missing {', '.join(missing)}"
