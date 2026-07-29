from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dlhub import artifacts
from dlhub.cli_utils import print_limited, summarize_output
from dlhub.zoo_registry import make_lazy_family_registry, split_arch_id


def test_split_arch_id_handles_bare_and_namespaced_ids() -> None:
    assert split_arch_id(" resnet18 ", default_prefix="DL") == ("dl", "resnet18")
    assert split_arch_id(" PC : pointnet ") == ("pc", "pointnet")
    with pytest.raises(ValueError, match="namespaced arch id"):
        split_arch_id("resnet18", example="dl:resnet18")
    with pytest.raises(ValueError, match="Invalid arch id"):
        split_arch_id("dl:")


def test_lazy_family_registry_centralizes_import_and_builder_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class Config:
        width: int

    calls: list[tuple[str, int, str]] = []

    def build_alpha(*, width: int, variant: str) -> object:
        calls.append(("alpha", width, variant))
        return object()

    module = SimpleNamespace(build_alpha=build_alpha)
    imported: list[str] = []

    def fake_import(name: str):
        imported.append(name)
        return module

    monkeypatch.setattr("dlhub.zoo_registry.importlib.import_module", fake_import)
    registry = make_lazy_family_registry(
        ["alpha"],
        ["tiny", "base"],
        module_template="example.{family}",
        builder_template="build_{family}",
        kwargs_factory=lambda cfg, variant: {"width": cfg.width, "variant": variant},
    )

    assert imported == []
    result = registry["alpha_tiny"](Config(width=16))
    assert result is not None
    assert imported == ["example.alpha"]
    assert calls == [("alpha", 16, "alpha_tiny")]


def test_shared_output_and_image_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    assert summarize_output({"mask": object(), "logits": object()}) == "dict(keys=[logits, mask])"
    assert summarize_output([object(), object(), object()]).endswith(", ... (+1)])")

    writes: list[tuple[object, object, int]] = []
    monkeypatch.setattr(
        artifacts,
        "_load_torchvision_save_image",
        lambda: lambda image, path, **kwargs: writes.append((image, path, kwargs["nrow"])),
    )
    image = object()
    path = tmp_path / "grid.png"
    assert artifacts.save_image_if_available(image, path, nrow=8)
    assert writes == [(image, path, 8)]


def test_shared_cli_limit_and_migrated_family_registry(capsys: pytest.CaptureFixture[str]) -> None:
    print_limited(["first", "middle", "last"], limit=2, tail=1)
    assert capsys.readouterr().out.splitlines() == ["first", "... (1 more) ...", "last"]

    from dlhub.vision.blur_detection_zoo import build_local_model, list_local_arches

    assert len(list_local_arches()) == 30
    model = build_local_model("blurdet:laplacian_blurdet_tiny", in_channels=3, width_mult=0.25)
    assert model.family == "laplacian_blurdet"
