from pathlib import Path

from scripts.narrative_check import check_zoo_claims


def test_zoo_claim_guard_rejects_implementation_count_language(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "zoo" / "example.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "70 个架构族 / 210 Architecture IDs\n所有实现均为纯 PyTorch\n",
        encoding="utf-8",
    )

    failures = check_zoo_claims((page,), root=tmp_path)

    assert len(failures) == 3
    assert all("ambiguous Zoo claim" in failure for failure in failures)


def test_zoo_claim_guard_accepts_registration_and_fidelity_language(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "tracks" / "example.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "70 个方法注册组 / 210 个注册 ID；源码机制以 fidelity 审计为准。\n",
        encoding="utf-8",
    )

    assert check_zoo_claims((page,), root=tmp_path) == []
