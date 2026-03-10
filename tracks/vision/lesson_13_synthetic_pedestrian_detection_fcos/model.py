from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "dldet:pedestrian_fcos"


def build_model(cfg: ModelConfig):
    raise NotImplementedError


__all__ = ["ModelConfig", "build_model"]

